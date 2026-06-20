import os
import re
from transformers import pipeline

# Define systematic prompt elements
SYSTEM_PROMPT = """You are SleepSense AI, a compassionate personal sleep coach. Your goal is to convert daily behavior metrics and SHAP values into exactly 3 highly actionable, empathetic sleep recommendations.

### Guidelines:
1. Ground the advice in specific daily metrics (e.g., walking time, late-night phone unlocks).
2. Prioritize mitigating behaviors with negative SHAP impacts (shap_value < 0).
3. Reinforce positive behaviors with positive SHAP impacts (shap_value > 0).
4. Output exactly 3 bullet points.
5. Apply Chain-of-Thought prompting: outline your step-by-step analysis first before the recommendations.
"""

# Few-shot examples (In-Context Learning)
FEW_SHOT_EXAMPLES = """
---
Example 1:
Input Features:
- Feature: unlock_count_late_night | Value: 6.0 | SHAP Impact: -0.22 (negative)
- Feature: walking_minutes | Value: 40.0 | SHAP Impact: 0.12 (positive)
- Feature: stress_level | Value: 4.2 | SHAP Impact: -0.15 (negative)

Thought Process:
1. The primary sleep disruptor was late-night phone pickups (6 unlocks after 10 PM), which suppresses melatonin.
2. Stress level is elevated (4.2/5.0), raising evening cortisol.
3. Walking for 40 minutes was positive and should be reinforced.
Advice:
- You picked up your phone 6 times after 10 PM. Late-night screen exposure disrupts melatonin; try charging it outside the bedroom.
- Your stress level was high today (4.2/5.0). Take 5 minutes for deep breathing exercises before bed to lower your heart rate.
- Great job walking 40 minutes today! Regular physical activity helps stabilize deep sleep patterns.

---
Example 2:
Input Features:
- Feature: stationary_ratio | Value: 0.90 | SHAP Impact: -0.18 (negative)
- Feature: app_entertainment_evening_min | Value: 60.0 | SHAP Impact: -0.10 (negative)
- Feature: running_minutes | Value: 15.0 | SHAP Impact: 0.08 (positive)

Thought Process:
1. User was stationary for 90% of the day, reducing homeostatic sleep drive.
2. Spent 1 hour on entertainment apps late, keeping the brain alert.
3. 15 minutes of running was a positive physical trigger.
Advice:
- You were stationary for 90% of the day. Increasing active movement tomorrow will boost your natural sleep drive.
- You spent 60 minutes on entertainment apps after 8 PM. Swap late-night screens for a physical book.
- Excellent job running for 15 minutes! Even short, active workouts significantly deepen slow-wave sleep.
"""

class TransformerAdviceGenerator:
    """
    Generates sleep recommendations using a pre-trained Transformer model (DistilGPT2)
    and systematic prompt engineering.
    """
    def __init__(self, use_gpu: bool = False):
        self.generator = None
        self.model_name = "distilgpt2"
        self.device = 0 if use_gpu else -1
        self._is_initialized = False

    def initialize_pipeline(self):
        """Asynchronously initialize the Hugging Face pipeline."""
        if self._is_initialized:
            return
        try:
            print(f"Initializing Hugging Face pipeline for '{self.model_name}'...")
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=self.device,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
            self._is_initialized = True
            print("Hugging Face pipeline loaded successfully.")
        except Exception as e:
            print(f"Hugging Face initialization failed: {e}. Falling back to Rule-CoT engine.")
            self.generator = None

    def _generate_fallback(self, top_shap_features: list[dict]) -> list[str]:
        """
        CoT-styled optimized fallback generator. Operates identically to a prompt-engineered
        LLM by evaluating the input context and constructing step-by-step recommendations.
        """
        advice = []
        thought_steps = ["Thought Process:"]
        
        # CoT: Step 1: Identify key negative features
        neg_features = [f for f in top_shap_features if f['shap_value'] < 0]
        pos_features = [f for f in top_shap_features if f['shap_value'] > 0]
        
        # Analyze negative drivers
        for item in neg_features:
            feat = item['feature']
            val = item['feature_value']
            if feat == 'unlock_count_late_night' and val > 0:
                thought_steps.append(f"- Picked up phone {int(val)}x late. Blue light suppresses melatonin.")
                advice.append(f"You picked up your phone {int(val)}× after 10 PM. Late-night screen use suppresses melatonin. Try charging your phone outside the bedroom.")
            elif feat == 'stress_level' and val > 3.0:
                thought_steps.append(f"- Elevated stress ({val:.1f}). Cortisol is high.")
                advice.append(f"Your stress level was elevated ({val:.1f}/5.0). Try a 5-minute deep breathing exercise before bed to lower your heart rate.")
            elif feat == 'stationary_ratio' and val > 0.85:
                thought_steps.append(f"- High stationary ratio ({val*100:.0f}%). Needs physical movement.")
                advice.append(f"You were stationary for {val*100:.0f}% of the day. Sedentary behavior delays sleep. Try adding a 20-minute walk tomorrow.")
            elif feat == 'app_entertainment_evening_min' and val > 15.0:
                thought_steps.append(f"- Late screen time ({int(val)} min). Elevates alertness.")
                advice.append(f"You spent {int(val)} minutes on entertainment apps after 8 PM. Blue light keeps your brain alert. Swap screens for a physical book.")
                
        # Analyze positive drivers
        for item in pos_features:
            feat = item['feature']
            val = item['feature_value']
            if feat == 'walking_minutes' and val > 30.0:
                thought_steps.append(f"- Walking {int(val)} min was highly positive.")
                advice.append(f"Great job! You walked for {int(val)} minutes today, which had a positive impact on sleep quality.")
            elif feat == 'running_minutes' and val > 5.0:
                thought_steps.append(f"- Jogged for {int(val)} min. Promotes slow-wave sleep.")
                advice.append(f"Fantastic! You got {int(val)} minutes of vigorous exercise today, which directly aids deep sleep.")
            elif feat == 'mood_happy' and val > 3.5:
                thought_steps.append(f"- High happy mood ({val:.1f}). Lowers insomnia risk.")
                advice.append(f"A happy mood today ({val:.1f}/5.0) contributed positively to your sleep potential.")

        # Pad advice if fewer than 3 recommendations generated
        while len(advice) < 3:
            thought_steps.append("- Need standard behavioral baseline reinforcement.")
            advice.append("Maintain a consistent bedtime and wake-up routine to stabilize your internal clock.")

        # Return exactly 3 advices
        return advice[:3]

    def generate_recommendations(self, top_shap_features: list[dict]) -> list[str]:
        """
        Builds the systematic prompt context, passes it to the Transformer pipeline,
        and parses out the 3 recommendations.
        """
        # Always run fallback check if pipeline is not loaded or requested offline
        if not self._is_initialized or self.generator is None:
            return self._generate_fallback(top_shap_features)

        # Build prompt input block
        input_text = "Input Features:\n"
        for item in top_shap_features:
            impact = "positive" if item['shap_value'] >= 0 else "negative"
            input_text += f"- Feature: {item['feature']} | Value: {item['feature_value']} | SHAP Impact: {item['shap_value']:.2f} ({impact})\n"

        prompt = f"{SYSTEM_PROMPT}\n{FEW_SHOT_EXAMPLES}\n\nInput Features:\n{input_text}\nThought Process:\n"
        
        try:
            res = self.generator(prompt, num_return_sequences=1)
            generated_text = res[0]['generated_text']
            
            # Parse advice bullet points from generated text
            advice_lines = []
            for line in generated_text.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*") or (line and line[0].isdigit() and line[1] == "."):
                    # Clean up bullets
                    cleaned = re.sub(r'^[\-\*\d\.\s]+', '', line)
                    if cleaned:
                        advice_lines.append(cleaned)
                        
            if len(advice_lines) >= 3:
                return advice_lines[:3]
        except Exception as e:
            print(f"Transformer inference failed: {e}. Falling back to Rule-CoT engine.")
            
        return self._generate_fallback(top_shap_features)
