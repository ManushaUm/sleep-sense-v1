import os
import re
import random
import traceback

class TransformerAdviceGenerator:
    """
    Generates sleep recommendations using the Google Gemini API (Free Tier)
    or a highly personalized, dynamic Rule-CoT fallback engine.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self._is_initialized = False

    def initialize_pipeline(self):
        """Initialize the Gemini client using API key from environment."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._is_initialized = True
                print("Gemini API client initialized successfully.")
            except Exception as e:
                print(f"Gemini API initialization failed: {e}. Falling back to dynamic Rule-CoT engine.")
                self._is_initialized = False
        else:
            print("GEMINI_API_KEY environment variable not found. Using dynamic Rule-CoT engine.")
            self._is_initialized = False

    def _generate_fallback(self, top_shap_features: list[dict], calendar_events: list[dict] = None) -> list[str]:
        """
        Generates clean, conversational, and highly personalized coaching advice based on the top SHAP features
        and tomorrow's calendar events. Utilizes randomized templates to provide maximum phrasing variety.
        """
        advice = []
        if calendar_events is None:
            calendar_events = []

        # -------------------------------------------------------------
        # Helper: Convert decimal hour to AM/PM string
        # -------------------------------------------------------------
        def format_hour(value):
            try:
                hr = int(value)
                min_val = int(round((value - hr) * 60))
                if min_val == 60:
                    hr += 1
                    min_val = 0
                ampm = "PM" if hr >= 12 else "AM"
                display_hr = hr % 12
                if display_hr == 0:
                    display_hr = 12
                return f"{display_hr}:{min_val:02d} {ampm}"
            except Exception:
                return f"{value:.1f}"

        # -------------------------------------------------------------
        # SLOT 1: Bedtime Target (Always Required - Calendar-Driven)
        # -------------------------------------------------------------
        earliest_event = None
        earliest_hour = None
        
        # Parse tomorrow's events to find the earliest morning commitment
        for ev in calendar_events:
            start_str = ev.get("start_time", "")
            # Expecting ISO format 'YYYY-MM-DDTHH:MM:SS' or similar. Extract hour
            try:
                # Extract time portion (HH:MM)
                time_match = re.search(r'T(\d{2}):(\d{2})', start_str)
                if time_match:
                    h = int(time_match.group(1))
                    m = int(time_match.group(2))
                    decimal_h = h + m / 60.0
                    
                    # Only focus on morning events (before 12 PM)
                    if decimal_h < 12.0:
                        if earliest_hour is None or decimal_h < earliest_hour:
                            earliest_hour = decimal_h
                            earliest_event = ev
            except Exception:
                pass

        if earliest_hour is not None:
            # bedtime = earliest_hour - 8hrs - 15min buffer (8.25 hours)
            target_h = (earliest_hour - 8.25) % 24
            formatted_bedtime = format_hour(target_h)
            event_time_str = format_hour(earliest_hour)
            event_title = earliest_event.get("summary", "commitment")
            
            slot1_options = [
                f"You have '{event_title}' tomorrow at {event_time_str}. Try to sleep by {formatted_bedtime} to secure a solid 8-hour sleep cycle.",
                f"With your '{event_title}' starting early at {event_time_str} tomorrow, target sleep by {formatted_bedtime} tonight so you wake up refreshed.",
                f"To prepare for your early '{event_title}' tomorrow morning at {event_time_str}, aim to wind down and sleep by {formatted_bedtime} tonight."
            ]
            advice.append(random.choice(slot1_options))
        else:
            # Free day tomorrow (no commitments before 12 PM)
            formatted_bedtime = "11:00 PM"
            slot1_options = [
                f"No early commitments on your schedule tomorrow! Capitalize on this free morning and sleep by {formatted_bedtime} to pay down any accumulated sleep debt.",
                f"Tomorrow morning looks open on your calendar. This is a great opportunity to sleep by {formatted_bedtime} and let your body naturally recover.",
                f"With a clear schedule tomorrow morning, aiming for sleep by {formatted_bedtime} will give you an excellent recovery opportunity."
            ]
            advice.append(random.choice(slot1_options))

        # -------------------------------------------------------------
        # SLOT 2: Behavioral Change (SHAP-Driven - Actionable)
        # -------------------------------------------------------------
        # Filter for negative shap drivers (what hurt sleep)
        neg_features = [f for f in top_shap_features if f.get('shap_value', 0) < 0]
        
        # Sort by absolute shap impact descending (worst driver first)
        neg_features = sorted(neg_features, key=lambda x: abs(x.get('shap_value', 0)), reverse=True)
        
        slot2_added = False
        for item in neg_features:
            feat = item.get('feature')
            val = item.get('feature_value', 0)
            
            if feat == 'unlock_count_late_night' and val > 0:
                options = [
                    f"I noticed you picked up your phone {int(val)} times late last night. Minimizing late-night screens helps trigger your body's natural melatonin release.",
                    f"Your phone was unlocked {int(val)} times late in the evening. Try leaving your device on your desk tonight to ease your mind into rest mode.",
                    f"It looks like you checked your screens {int(val)} times late today. Swap late-night scrolling for a book to prepare your eyes for sleep."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat == 'last_unlock_hour' and val >= 22.0:
                formatted_val = format_hour(val)
                options = [
                    f"Your final screen check was at {formatted_val} today. Try setting a 'device-free' wind-down routine 30 minutes before sleep.",
                    f"Your last phone pickup was late at {formatted_val}. Putting your phone face-down from 10 PM will help lower evening stimulation.",
                    f"With your last screen usage at {formatted_val}, allocating a device-free transition period before bed will significantly improve your sleep onset."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat in ['app_entertainment_evening_min', 'app_late_night_min'] and val > 0.0:
                options = [
                    f"You spent {int(val)} minutes on entertainment or social apps late. Try swapping late-night screen time for reading a book or listening to soft music.",
                    f"I noticed you spent {int(val)} minutes scrolling apps late tonight. Switch to audiobooks or soft music to let your eyes recover from light.",
                    f"Late app scrolling ({int(val)} mins) stimulates cognitive alertness. Dim your screens and wrap up phone usage by 10 PM."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat == 'stress_level' and val >= 3.0:
                options = [
                    f"With your reported stress level elevated ({val:.1f}/5.0) today, consider taking 5 minutes to write down your worries or make a to-do list.",
                    f"I saw you experienced a higher-tension day today ({val:.1f}/5.0). Doing a quick breathing exercise before bedtime will help lower evening cortisol.",
                    f"Since today felt a bit stressful ({val:.1f}/5.0), spend 10 minutes relaxing screen-free before bed to clear cognitive tension."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat == 'stationary_ratio' and val > 0.80:
                options = [
                    f"You were inactive for {int(val*100)}% of the day. A sedentary routine reduces sleep pressure; try adding a short, active walk tomorrow.",
                    f"I noticed you were stationary for {int(val*100)}% of today. Plan a brisk 15-minute outdoor walk tomorrow to build healthy physical tiredness.",
                    f"Increasing active movement tomorrow will boost your natural sleep drive after a highly sedentary day ({int(val*100)}% stationary)."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat == 'avg_session_duration_min' and val > 5.0:
                options = [
                    f"Your average screen session was quite long ({val:.1f} minutes). Taking frequent screen breaks throughout the day helps lower cognitive fatigue.",
                    f"Taking shorter, more focused screen breaks tomorrow will help your brain stay relaxed and improve sleep stability."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat == 'screen_sessions_count' and val > 30.0:
                options = [
                    f"You checked your phone {int(val)} times today. High screen interaction frequency can keep your mind stimulated. Try setting screen-free focus hours tomorrow.",
                    f"With {int(val)} phone pickups today, reducing screen triggers tomorrow will lower mental fatigue and help you fall asleep easier."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break
            elif feat in ['nlp_caffeine_similarity', 'nlp_stress_similarity', 'nlp_screen_similarity'] and val > 0.5:
                options = [
                    f"Your journal notes highlighted potential sleep triggers (like caffeine or anxiety). Consider writing down tomorrow's tasks to clear your mind.",
                    f"Reflecting on today's triggers in your journal is a great start. Swap caffeine and screens for a relaxing bedtime tea to prepare for rest."
                ]
                advice.append(random.choice(options))
                slot2_added = True
                break

        # Fallback Slot 2 if no negative driver met thresholds
        if not slot2_added:
            # Check for a positive driver to reinforce
            pos_features = [f for f in top_shap_features if f.get('shap_value', 0) > 0]
            pos_features = sorted(pos_features, key=lambda x: abs(x.get('shap_value', 0)), reverse=True)
            
            for item in pos_features:
                feat = item.get('feature')
                val = item.get('feature_value', 0)
                
                if feat == 'walking_minutes' and val >= 20.0:
                    options = [
                        f"Great job walking for {int(val)} minutes today! Keeping active in daylight helps reinforce your natural sleep-wake rhythm.",
                        f"Your walking time today ({int(val)} mins) was excellent. Daily physical activity is a fantastic trigger for deep sleep cycles."
                    ]
                    advice.append(random.choice(options))
                    slot2_added = True
                    break
                elif feat == 'running_minutes' and val >= 10.0:
                    options = [
                        f"Excellent work getting {int(val)} minutes of cardiovascular running today! This helps deepen slow-wave sleep cycles.",
                        f"A good running session of {int(val)} minutes will work wonders for your deep sleep recovery tonight. Keep it up!"
                    ]
                    advice.append(random.choice(options))
                    slot2_added = True
                    break
                elif feat in ['exercise_detected', 'exercise_self_report'] and val > 0.5:
                    advice.append("Filing an exercise log today was a great choice. Physical activity plays a major role in expanding deep slow-wave sleep cycles.")
                    slot2_added = True
                    break
            
            # Baseline fallback Slot 2 if still empty
            if not slot2_added:
                options = [
                    "To support healthy sleep, establish a screen-free wind-down routine starting 30 minutes before your bedtime target.",
                    "Keep your phone out of arm's reach tonight. Charging it across the room reduces the temptation for late-night scrolling.",
                    "Consider replacing evening screen interactions with light stretching or listening to relaxing music."
                ]
                advice.append(random.choice(options))

        # -------------------------------------------------------------
        # SLOT 3: Tomorrow Preparation (Calendar-Driven)
        # -------------------------------------------------------------
        HIGH_STRESS_KEYWORDS = ["exam", "test", "quiz", "deadline", "presentation", "interview", "submission", "viva", "defense", "review"]
        LATE_SOCIAL_KEYWORDS = ["party", "dinner", "meetup", "hangout", "event", "concert", "match", "game", "gathering"]
        PHYSICAL_KEYWORDS = ["gym", "training", "practice", "sport", "workout", "run"]

        # Classify the primary event tomorrow
        primary_category = "normal"
        primary_event_title = ""
        
        # Check all tomorrow's events
        for ev in calendar_events:
            title = ev.get("summary", "").lower()
            if any(k in title for k in HIGH_STRESS_KEYWORDS):
                primary_category = "high_stress"
                primary_event_title = ev.get("summary", "important task")
                break # High stress takes absolute priority
            elif any(k in title for k in LATE_SOCIAL_KEYWORDS):
                primary_category = "late_social"
                primary_event_title = ev.get("summary", "event")
            elif any(k in title for k in PHYSICAL_KEYWORDS):
                if primary_category == "normal":
                    primary_category = "physical"
                    primary_event_title = ev.get("summary", "workout")

        if primary_category == "high_stress":
            options = [
                f"You have '{primary_event_title}' tomorrow. Avoid studying after 11 PM — sleep consolidates memory and learning better than last-minute cramming.",
                f"For your '{primary_event_title}' tomorrow, take 5-10 minutes for slow deep breathing before bed to lower cortisol and speed up sleep onset.",
                f"Prepare your materials for '{primary_event_title}' tonight so your morning starts low-effort, low-anxiety, and structured."
            ]
            advice.append(random.choice(options))
        elif primary_category == "late_social":
            options = [
                f"You have a late social event ('{primary_event_title}') scheduled. Post-event adrenaline keeps you alert; plan to start your wind-down immediately after returning.",
                f"With '{primary_event_title}' scheduled late, avoid screen scrolling immediately after returning home to help clear neural adrenaline.",
                f"Social events can shift bedtime later. Avoid drinking caffeine late during '{primary_event_title}' to protect your sleep architecture."
            ]
            advice.append(random.choice(options))
        elif primary_category == "physical":
            options = [
                f"With your '{primary_event_title}' workout tomorrow, prioritize high sleep quality tonight to facilitate optimal muscle recovery and cellular repair.",
                f"Hydrating well tonight will prepare your body for tomorrow's physical workout ('{primary_event_title}') and protect sleep quality."
            ]
            advice.append(random.choice(options))
        else:
            # Normal category
            options = [
                "Avoid eating heavy meals or consuming caffeine within 4-6 hours of bed to prevent light, fragmented sleep cycles.",
                "Ensure your bedroom remains cool, dark, and quiet tonight to help stabilize your circadian rhythm.",
                "Stick strictly to your target bedtime tonight and avoid the temptation to scroll on your phone in bed."
            ]
            advice.append(random.choice(options))

        return advice[:3]

    def generate_recommendations(self, top_shap_features: list[dict], calendar_events: list[dict] = None) -> list[str]:
        """
        Generates SleepSense coaching advice using Gemini 1.5 Flash API,
        with a fully dynamic Rule-CoT fallback engine on error.
        """
        if calendar_events is None:
            calendar_events = []

        # Force initialize if not done yet
        if not self._is_initialized and self.api_key:
            self.initialize_pipeline()

        # If initialized, query Gemini
        if self._is_initialized:
            try:
                import google.generativeai as genai
                
                # Format metrics and SHAP values for the prompt
                shap_details = []
                for item in top_shap_features:
                    feat = item.get('feature')
                    val = item.get('feature_value', 0.0)
                    shap_val = item.get('shap_value', 0.0)
                    impact = "positive (helps sleep)" if shap_val > 0 else "negative (disrupts sleep)"
                    shap_details.append(f"- Feature: {feat} | Value: {val} | SHAP Impact: {shap_val:.4f} ({impact})")
                shap_details_str = "\n".join(shap_details) if shap_details else "No direct telemetry drivers recorded today."

                # Format calendar events for the prompt
                calendar_details = []
                for ev in calendar_events:
                    summary = ev.get("summary", "Event")
                    start = ev.get("start_time", "Unknown")
                    end = ev.get("end_time", "Unknown")
                    calendar_details.append(f"- Event: {summary} | Starts: {start} | Ends: {end}")
                calendar_details_str = "\n".join(calendar_details) if calendar_details else "No scheduled events tomorrow."

                # Construct prompt strictly enforcing your calander-recommedation-guide rules
                prompt = f"""You are SleepSense AI, a compassionate personal sleep coach. Convert the following daily behavior metrics, SHAP values, and tomorrow's calendar schedule into exactly 3 highly actionable, empathetic sleep recommendations.

### Guidelines:
1. Ground the advice directly in the daily metrics and tomorrow's schedule.
2. Output EXACTLY 3 bullet points starting with "- ". Do not include intro, outro, headers, or general chatter.
3. Keep the tone warm, conversational, and personalized. Avoid dry system-log style text.
4. Follow the specific Slots structure:
   - Slot 1: Target Bedtime. Find the earliest morning event starting before 12 PM and recommend sleep (bedtime = event_start_time - 8 hours - 15 min buffer). If no early event, recommend sleep by 11:00 PM to recover sleep debt. Include a specific time and reason.
   - Slot 2: Behavioral Advice. Target the worst sleep disruptor (lowest SHAP value) with a concrete, physical action.
   - Slot 3: Tomorrow Preparation. Focus on tomorrow's key calendar events (e.g. exams, presentations, social events). Adjust advice based on event stress/type.

Input Data:
Today's Top Telemetry Drivers (SHAP):
{shap_details_str}

Tomorrow's Calendar Events:
{calendar_details_str}
"""
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                text = response.text.strip()
                # Parse bullet points into a list of strings
                bullets = [line.strip().replace("- ", "", 1).strip() for line in text.split("\n") if line.strip().startswith("-")]
                
                # Verify we got exactly 3 bullets, otherwise raise exception to trigger fallback
                if len(bullets) == 3:
                    return bullets
                else:
                    raise ValueError(f"Gemini returned invalid number of bullets: {len(bullets)}")

            except Exception as e:
                print("Gemini API call failed or timed out:")
                traceback.print_exc()
                print("Falling back to local dynamic Rule-CoT engine.")

        # Fallback to local rule engine
        return self._generate_fallback(top_shap_features, calendar_events)
