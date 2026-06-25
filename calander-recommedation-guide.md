# Calendar-Aware Sleep Recommendations — SleepSense

> **Design Principle:** Every recommendation must be grounded in two signals — _what the user did today_ (SHAP behavioral drivers) and _what is coming tomorrow_ (calendar events). Generic advice is explicitly forbidden.

---

## 1. How Calendar Events Map to Recommendations

### 1.1 Early Morning Events → Bedtime Enforcement

**Trigger:** Any event starting before 9:00 AM tomorrow.

**Logic:**

- Required sleep = 7–8 hours
- Target bedtime = event start time − 8 hours
- Buffer = 15 min (falling asleep lag)

**Example Recommendations:**

- "You have a 9AM lecture tomorrow. Target sleep by 10:45PM tonight to get 8 full hours."
- "Set a phone-down alarm for 10PM — your body needs 30–45 min to wind down before your target bedtime."

---

### 1.2 High-Stress Events → Pre-Sleep Wind Down

**Trigger:** Keywords in event title matching: `exam`, `test`, `quiz`, `deadline`, `presentation`, `interview`, `submission`, `viva`.

**Logic:**

- High-stress events elevate cortisol, which delays sleep onset
- Last-minute studying after 11PM impairs memory consolidation more than it helps retention
- Pre-sleep breathing/relaxation lowers cortisol before bed

**Example Recommendations:**

- "You have an exam tomorrow. Avoid studying after 11PM — sleep consolidates what you already know far better than last-minute cramming."
- "Do 10 minutes of slow breathing or light stretching before bed to lower cortisol and speed up sleep onset."
- "Prepare your exam materials tonight so tomorrow morning is low-effort and low-anxiety."

---

### 1.3 Late Night Events → Adrenaline Recovery Warning

**Trigger:** Any event ending after 8:00 PM.

**Logic:**

- Post-event adrenaline (presentations, social events, sports) peaks 1–2 hours after activity
- Sleep onset realistically shifts 2 hours later
- Next-morning commitments should be checked for conflict

**Example Recommendations:**

- "Your presentation finishes at 8PM. Post-event adrenaline typically keeps you alert until 10–11PM — plan for a 11:30PM sleep at the earliest."
- "Avoid screens after your event; wind down with music or reading instead to accelerate adrenaline clearance."

---

### 1.4 Back-to-Back High-Load Days → Sleep Debt Warning

**Trigger:** 3 or more events detected across the next 2 days OR multiple high-stress events in the same week.

**Logic:**

- Accumulated sleep debt compounds cognitive impairment
- Each night of <6 hours increases next-night sleep pressure
- One full-length sleep night significantly recovers performance

**Example Recommendations:**

- "You have 3 deadlines this week. Prioritize 8 hours minimum tonight — even one short night now will compound fatigue by Thursday."
- "Sleep debt is building. Protect your sleep window tonight even if it means stopping work earlier than planned."

---

### 1.5 Free Day Tomorrow → Recovery Opportunity Signal

**Trigger:** No events before 12:00 PM on the next day.

**Logic:**

- Low next-morning pressure removes the cost of a longer sleep window
- Evening exercise is safe tonight since recovery time is available
- This is the best night to reclaim sleep debt

**Example Recommendations:**

- "No early commitments tomorrow — this is your best opportunity this week to get 9 hours and recover sleep debt."
- "Safe to do light exercise tonight; physical tiredness from activity deepens slow-wave sleep, and you have time to recover tomorrow."

---

## 2. Combining SHAP Drivers + Calendar (Core Intelligence)

The most powerful recommendations come from fusing behavioral data with schedule context. Neither signal alone is sufficient.

### Decision Matrix

| SHAP Top Disruptor            | Calendar Signal         | Combined Recommendation                                                                                                           |
| ----------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| High late-night phone unlocks | Early event tomorrow    | "You've been picking up your phone late AND have an 8AM lab. Phone down by 10PM is critical tonight."                             |
| Good walking activity today   | Free day tomorrow       | "Great activity today builds sleep pressure — capitalize on this with an early bedtime since you have no early commitments."      |
| High stress EMA score         | Exam tomorrow           | "You're already stressed today AND have an exam tomorrow. Stop all work by 10:30PM — sleep is your best exam prep now."           |
| Low activity today            | Heavy schedule tomorrow | "Low movement today means less physical sleep drive tonight. Avoid caffeine after 3PM and stick strictly to your bedtime target." |
| Late app usage (social)       | Presentation tomorrow   | "Late social app use delays melatonin production. With a presentation tomorrow, switch to Do Not Disturb mode by 10PM."           |

---

## 3. Event Stress Classification

Classify incoming calendar events before generating advice:

```python
HIGH_STRESS_KEYWORDS = [
    "exam", "test", "quiz", "deadline", "presentation",
    "interview", "submission", "viva", "defense", "review"
]

LATE_SOCIAL_KEYWORDS = [
    "party", "dinner", "meetup", "hangout", "event",
    "concert", "match", "game", "gathering"
]

PHYSICAL_KEYWORDS = [
    "gym", "training", "practice", "sport", "workout", "run"
]

def classify_event_stress(event_title: str) -> str:
    title = event_title.lower()
    if any(k in title for k in HIGH_STRESS_KEYWORDS):
        return "high_stress"
    elif any(k in title for k in LATE_SOCIAL_KEYWORDS):
        return "late_social"
    elif any(k in title for k in PHYSICAL_KEYWORDS):
        return "physical"
    else:
        return "normal"
```

---

## 4. The Three Recommendation Slots

Every Sleep Coach response must fill exactly three slots. No more, no less.

### Slot 1 — Bedtime Target (Always Required)

- **Input:** Earliest calendar event tomorrow
- **Formula:** `bedtime = earliest_event_time − 8hrs − 15min_buffer`
- **Format:** Specific time + one-line reasoning
- **Example:** "Target sleep by 10:45PM — your 7AM lab needs 8 hours of rest."

### Slot 2 — Behavioral Change (SHAP-Driven)

- **Input:** User's single worst SHAP disruptor today
- **Rule:** Must reference the specific behavior, not a generic tip
- **Example:** "You had 14 late-night phone pickups today. Set your phone face-down on your desk — not beside your bed — from 10PM."

### Slot 3 — Tomorrow Preparation (Calendar-Driven)

- **Input:** Classified stress level of the most important event tomorrow
- **Rule:** Must be specific to the event type
- **Example (high_stress):** "Exam tomorrow — avoid heavy meals after 8PM as digestion fragments sleep architecture."
- **Example (late_social):** "Late event tonight — set an alarm not just for waking but for starting your wind-down routine."
- **Example (normal):** "Standard day tomorrow — stick to your bedtime target and avoid the temptation to scroll in bed."
