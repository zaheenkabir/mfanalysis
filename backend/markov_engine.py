import numpy as np
import pandas as pd

class MarkovAnalyzer:
    """
    Analyzes historical performance using Markov Chain Transition Matrices
    to determine regime reliability and alpha persistence.
    """
    
    @staticmethod
    def calculate_reliability_score(fund_nav, bench_nav):
        """
        Calculates regime reliability using rolling monthly returns (21 days).
        Returns a dictionary with score, transition matrix, and state counts.
        """
        if len(fund_nav) < 60 or len(bench_nav) < 60:
            return {'score': 50.0, 'debug': 'Insufficient History'}
            
        # 1. Align Data
        df = pd.DataFrame({'fund_price': fund_nav, 'bench_price': bench_nav}).dropna()
        if len(df) < 60: return {'score': 50.0}
        
        # 2. Calculate Rolling Monthly Returns (21 days)
        # We use rolling windows to reduce daily noise and capture actual "trends"
        df['fund_roll'] = df['fund_price'].pct_change(21) * 100
        df['bench_roll'] = df['bench_price'].pct_change(21) * 100
        df = df.dropna()
        
        # 3. Calculate Excess Returns (Alpha)
        excess = df['fund_roll'] - df['bench_roll']
        
        # 4. Discretize into States
        # Threshold: 0.5% monthly alpha (~6% annualized alpha)
        # State 0: Lagging (<-0.5%)
        # State 1: Matching (-0.5% to 0.5%)
        # State 2: Leading (>0.5%)
        states = np.zeros(len(excess), dtype=int)
        states[excess < -0.5] = 0
        states[(excess >= -0.5) & (excess <= 0.5)] = 1
        states[excess > 0.5] = 2
        
        # 5. Build Transition Matrix
        n_states = 3
        tpm = np.zeros((n_states, n_states))
        for (i, j) in zip(states, states[1:]):
            tpm[i][j] += 1
            
        # Normalize
        row_sums = tpm.sum(axis=1, keepdims=True)
        probs = np.divide(tpm, row_sums, out=np.zeros_like(tpm), where=row_sums!=0)
        
        # 6. Metrics
        persistence = probs[2, 2] * 100 # Prob of staying Winner
        recovery = probs[0, 2] * 100    # Prob of Lagging -> Leading (Bonus)
        
        # Calculate Steady State (Long term probability of being in State 2)
        try:
            steady = np.linalg.matrix_power(probs, 50)
            steady_prob = steady[0, 2] * 100
        except:
            steady_prob = persistence

        # Score Weighting:
        # 70% Persistence (Can you keep winning?)
        # 30% Steady State (Are you winning often in the long run?)
        score = (persistence * 0.7) + (steady_prob * 0.3)
        
        return {
            'score': round(float(score), 1),
            'tpm': probs,
            'steady_state': steady_prob,
            'persistence': persistence,
            'states': states,
            'counts': np.bincount(states, minlength=3)
        }

    @staticmethod
    def classify_fund_persona(reliability_score):
        """Returns a user-friendly classification (Persona) based on the score."""
        if reliability_score >= 80:
            return "🛡️ Consistent Compounder"
        elif reliability_score >= 65:
            return "📈 Steady Performer"
        elif reliability_score >= 50:
            return "🔄 Cyclical / Tactical"
        else:
            return "⚠️ Volatile / Unpredictable"

    @staticmethod
    def get_regime_insight(reliability_score):
        """Returns human-readable insight based on Markov reliability."""
        if reliability_score >= 70:
            return "This fund is a winning machine. It rarely underperforms and tends to stay in the 'Leading' zone for long periods. A great 'Fill it, Shut it, Forget it' choice."
        elif reliability_score >= 50:
            return "This fund has its moments. It generally performs well but can go through dull phases. Best used as a diversifying satellite holding rather than your core portfolio."
        else:
            return "This fund is quite unpredictable. It might give huge returns one month and crash the next. Suitable only if you time your entry and exit perfectly."
