import numpy as np
import pandas as pd
from markov_engine import MarkovAnalyzer

def get_category_adjustments(category):
    """Returns baseline expectations for different fund categories based on institutional indices."""
    cat = category.lower() if category else ""
    if 'small' in cat:
        return {'alpha_target': 8, 'beta_target': 1.15, 'mdd_tol': 35, 'dc_target': 95}
    elif 'mid' in cat:
        return {'alpha_target': 6, 'beta_target': 1.05, 'mdd_tol': 25, 'dc_target': 90}
    elif 'flexi' in cat or 'multi' in cat:
        return {'alpha_target': 5, 'beta_target': 1.0, 'mdd_tol': 22, 'dc_target': 85}
    elif 'elss' in cat or 'tax' in cat:
        return {'alpha_target': 5, 'beta_target': 1.0, 'mdd_tol': 22, 'dc_target': 90}
    elif 'hybrid' in cat or 'balanced' in cat:
        return {'alpha_target': 3, 'beta_target': 0.7, 'mdd_tol': 15, 'dc_target': 75}
    elif 'debt' in cat or 'liquid' in cat:
        return {'alpha_target': 1.5, 'beta_target': 0.3, 'mdd_tol': 4, 'dc_target': 40}
    return {'alpha_target': 4, 'beta_target': 1.0, 'mdd_tol': 20, 'dc_target': 90} # Default/Large Cap

def calculate_fund_score(metrics, rolling_stats=None, mc_stats=None, category=None, multi_metrics=None):
    """
    Calculates a weighted composite score (0-100) using a Multi-Factor model.
    multi_metrics: dict of {period: metrics_dict} for 1Y, 3Y, 5Y stability.
    """
    markov_data = metrics.get('markov_reliability', {})
    if isinstance(markov_data, dict):
        reliability_score = markov_data.get('score', 50.0)
    else:
        reliability_score = float(markov_data)
        
    adj = get_category_adjustments(category or metrics.get('Category', ''))
    
    # helper for single period score
    def get_period_score(m, r=None):
        p_scores = []
        s_count = 0
        
        # Performance
        a = m.get('Alpha', 0)
        shp = m.get('Sharpe Ratio', 0)
        srt = m.get('Sortino Ratio', 0)
        pts = min(20, max(0, (a / adj['alpha_target']) * 20)) + min(12, max(0, shp * 15)) + min(8, max(0, srt * 10))
        if a > adj['alpha_target']: s_count += 1
        p_scores.append({'s': pts, 'm': 40})
        
        # Risk (Institutional Grade: Normalized by Category Target)
        b = m.get('Beta', 1.0)
        dc = m.get('Downside Capture', 100)
        mx = abs(m.get('Max Drawdown', 0))
        
        # Beta Score: Target category beta as baseline (institutional grade)
        # Use a steeper penalty for high beta (> baseline + 0.3)
        excess_beta = max(0, b - (adj['beta_target'] + 0.1))
        b_pts = max(0, 10 - (excess_beta * 20)) # Linear but steep penalty
        
        # Downside Capture Score: Relative to dc_target
        # Penalty increases if > target, bonus if << target
        if dc > adj['dc_target']:
            # Non-linear penalty for poor downside protection
            dc_pts = max(0, 10 - ((dc - adj['dc_target']) / 2.5) ** 1.2)
        else:
            dc_pts = min(12, 10 + (adj['dc_target'] - dc) / 4) # Slightly more reward for elite protection
            
        mdd_pts = max(0, min(8, (adj['mdd_tol'] * 1.5 - mx) / 2))
        
        r_pts = b_pts + dc_pts + mdd_pts
        if dc < adj['dc_target'] * 0.85: s_count += 2
        p_scores.append({'s': r_pts, 'm': 30})
        
        return sum(x['s'] for x in p_scores), sum(x['m'] for x in p_scores), s_count

    # Multi-window stability check
    final_earned = 0
    final_possible = 0
    total_synergy = 0
    
    if multi_metrics and len(multi_metrics) > 1:
        # Weights: 5Y: 50%, 3Y: 30%, 1Y: 20% (Stability focus)
        weights = {'5Y': 0.5, '3Y': 0.3, '1Y': 0.2}
        valid_weight_sum = 0
        for period, weight in weights.items():
            if period in multi_metrics:
                s, m, syn = get_period_score(multi_metrics[period])
                final_earned += (s / m) * 100 * weight
                valid_weight_sum += weight
                total_synergy += syn
        
        if valid_weight_sum > 0:
            base_score = final_earned / valid_weight_sum
        else:
            s, m, total_synergy = get_period_score(metrics)
            base_score = (s / m) * 100
    else:
        s, m, total_synergy = get_period_score(metrics)
        base_score = (s / m) * 100

    # Fad Penalty: If 1Y alpha is > 2.5x of 3Y alpha (potential luck)
    if multi_metrics and '1Y' in multi_metrics and '3Y' in multi_metrics:
        alpha_1y = multi_metrics['1Y'].get('Alpha', 0)
        alpha_3y = multi_metrics['3Y'].get('Alpha', 0)
        if alpha_1y > 10 and alpha_1y > (alpha_3y * 2.5):
            base_score -= 12 # Increased fad penalty

    # Pillar: Consistency & Reliability
    extra_pts = 0
    possible_extra = 0
    
    # Consistency Pillar (Standardized to 3Y rolling if possible)
    if rolling_stats and 'outperformance_pct' in rolling_stats:
        op = rolling_stats.get('outperformance_pct', 0)
        extra_pts += (op / 100) * 20
        possible_extra += 20
        if op > 80: total_synergy += 2 # Tightened synergy trigger
        
    if mc_stats and 'prob_success_15' in mc_stats:
        p15 = mc_stats.get('prob_success_15', 0)
        extra_pts += (p15 / 100) * 10
        possible_extra += 10

    # Normalize base score (60% weight) + Extras (25% weight) + Markov (15% weight)
    final_score = (base_score * 0.6) + (extra_pts / possible_extra * 25 if possible_extra > 0 else 12.5) + (reliability_score / 100 * 15)
    
    # Synergy Bonus (Elite Institutional Alpha)
    # 1. Broad Excellence
    if total_synergy >= 6: final_score += 3 # Harder to get
    if total_synergy >= 9: final_score += 5 
    # 2. Risk-Adjusted Alpha (High Alpha + Low Beta/DC)
    if metrics.get('Alpha', 0) > (adj['alpha_target'] + 2) and metrics.get('Downside Capture', 100) < (adj['dc_target'] - 5):
        final_score += 4
    
    return min(100, max(0, final_score))

def get_verdict_metadata(score, reliability=50):
    if score >= 85: 
        msg = "Simply the best. This fund is an elite wealth creator."
        if reliability >= 70: msg += " It's a consistent winner you can trust."
        return "Platinum Elite Buy", "💎", "green", msg
    elif score >= 75: 
        return "Strong Buy", "🌟", "green", "Top-tier quality. Consistent returns with solid safety."
    elif score >= 65: 
        return "Quality Buy", "✅", "blue", "A good, reliable fund for your core portfolio."
    elif score >= 45: 
        return "Neutral / Hold", "⚖️", "orange", "Decent performance, but nothing extraordinary. Compare with peers."
    else:
        return "Avoid / Caution", "⚠️", "red", "Risky or underperforming. Better options exist."

def generate_ml_insights(metrics, score):
    """
    Simulates ML behavior by analyzing metric outliers and confidence.
    """
    alpha = metrics.get('Alpha', 0)
    sharpe = metrics.get('Sharpe Ratio', 0)
    beta = metrics.get('Beta', 1.0)
    
    # Confidence Score based on parameter stability
    # If Alpha is high but Sharpe is low, confidence is lower (luck? high risk?)
    confidence = 100
    if alpha > 5 and sharpe < 0.5: confidence -= 30
    if beta > 1.5: confidence -= 20
    
    # Factor Analysis
    primary_factor = "Alpha Generation" if alpha > 3 else "Risk Management" if beta < 0.9 else "Beta Exposure"
    
    return {
        "confidence": max(50, confidence),
        "primary_factor": primary_factor,
        "is_outlier": alpha > 10 or alpha < -5
    }

def generate_single_verdict(metrics, rolling_stats=None, mc_stats=None, category=None, multi_metrics=None):
    score = calculate_fund_score(metrics, rolling_stats, mc_stats, category, multi_metrics)
    
    markov_data = metrics.get('markov_reliability', {})
    if isinstance(markov_data, dict):
        reliability_score = markov_data.get('score', 50.0)
    else:
        reliability_score = float(markov_data)

    label, icon, color, text = get_verdict_metadata(score, reliability_score)
    
    ml_data = generate_ml_insights(metrics, score)
    # Add fad alert to ML insights
    if multi_metrics and '1Y' in multi_metrics and '3Y' in multi_metrics:
        if multi_metrics['1Y'].get('Alpha', 0) > (multi_metrics['3Y'].get('Alpha', 0) * 2.5):
            ml_data['fad_warning'] = True
            text = "⚠️ CAUTION: High recent returns may be luck. " + text
    
    pillar_scores = {
        "Stability": "High" if score > 75 else "Moderate" if score > 55 else "Low",
        "Risk Control": "Superior" if metrics.get('Beta', 1.0) < 0.85 else "Strong" if metrics.get('Beta', 1.0) < 1.05 else "Aggressive"
    }
    
    return {
        "score": round(score, 1),
        "label": label,
        "icon": icon,
        "color": color,
        "verdict_text": text,
        "pillar_scores": pillar_scores,
        "ml_insights": ml_data
    }

def generate_comparison_verdict_advanced(funds_scores):
    """
    Analyzes pre-calculated scores to determine the best choice among peers.
    funds_scores: dict of {name: score}
    """
    if not funds_scores: return None
        
    winner = max(funds_scores, key=funds_scores.get)
    margin = funds_scores[winner] - (sum(funds_scores.values()) / len(funds_scores))
    
    reasoning = f"{winner} leads with {funds_scores[winner]:.1f}/100. "
    if margin > 10:
        reasoning += "It shows a significant performance edge over its peers."
    else:
        reasoning += "It offers a slightly better risk-reward profile in the current set."
        
    return {
        "winner": winner,
        "all_scores": funds_scores,
        "reasoning": reasoning
    }

def generate_portfolio_verdict(portfolio_metrics, rolling_stats=None, mc_stats=None, category="Portfolio"):
    score = calculate_fund_score(portfolio_metrics, rolling_stats, mc_stats, category)
    label, icon, color, text = get_verdict_metadata(score)
    
    # Portfolio pillars
    pillar_scores = {
        "Efficiency (Risk/Return)": "High" if score > 75 else "Moderate" if score > 55 else "Low",
        "Consistency": "Stable" if (rolling_stats and rolling_stats.get('outperformance_pct', 0) > 65) else "Variable",
        "Goal Probability": f"{mc_stats.get('prob_success_15', 0):.0f}%" if mc_stats else "N/A"
    }

    return {
        "score": score,
        "label": f"Portfolio {label}",
        "icon": "🏥" if score < 50 else "🎯",
        "color": color,
        "verdict_text": text,
        "pillar_scores": pillar_scores,
        "health_index": score
    }
