import os
import json
import openai
from dotenv import load_dotenv
load_dotenv()  # this will load OPENAI_API_KEY from .env


"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

What I would have done if I had spent 2 more hours on this project is that I would have included the following:

1. Add a safety and reading level analyzer (e.g. controlling vocabulary difficulty, sentence length, and emotional intensity). This would make sure that stories are not only age-appropriate, but also tailored for kids' (ages 5-10) exact reading level.

2. Add a theme/style selector and preset story modes such as "adventure", "friendship", or "sleepy-time", which would shape the tone and improve personalization.

3. Add a small local web-interface for easier interaction instead of using the console/terminal by building a Flask or Streamlit UI where users can generate their stories, apply their refinements, and continue editing interactively. This would ensure deployment readiness and improve UX.

These additions would make the system more robust, user-friendly, and closer to building a multi-agent storytelling tool.

"""


def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
    # please use your own openai api key here.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message["content"]  # type: ignore

# example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

# Story writer (first agent) – creates the initial bedtime story draft.
STORY_WRITER_PROMPT = """
You are an expert children's author.

Write a bedtime story for a child aged between 5 and 10 years old.

Requirements:
- The story must be gentle, kind, and emotionally safe.
- Use simple, concrete language that a 2nd–4th grader can understand.
- Length: about 350–600 words.
- Clear beginning, middle, and end.
- A small, non-scary problem that gets resolved positively.
- Include a short 1–2 sentence moral at the end starting with "Moral:".

The child's story request is:
"{user_request}"

Now write ONLY the story. Do not add any explanation or commentary.
"""

# Judge (second agent) – evaluates a story and returns structured JSON feedback.
JUDGE_PROMPT = """
You are a strict but kind editorial judge for children's bedtime stories (ages 5–10).

Your job is to review the story below and return a JSON object ONLY, with no extra text.

Evaluate the story on:
- clarity (0–10)
- age_appropriateness (0–10)
- emotional_safety (0–10)
- creativity (0–10)
- narrative_structure (0–10)

Also include:
- total_score: sum of the five scores
- issues: a short list of strings describing concrete problems
- suggested_fixes: a short list of specific, actionable improvements

Return JSON in this format ONLY:
{{
  "clarity": 9,
  "age_appropriateness": 10,
  "emotional_safety": 9,
  "creativity": 8,
  "narrative_structure": 9,
  "total_score": 45,
  "issues": ["issue 1", "issue 2"],
  "suggested_fixes": ["fix 1", "fix 2"]
}}

Here is the story to judge:
\"\"\"{story}\"\"\"
"""

# Revision agent (third agent) – rewrites the story to address judge feedback.
REVISION_PROMPT = """
You are revising a children's bedtime story for ages 5–10.

Original story:
\"\"\"{story}\"\"\"

Feedback from a quality judge:

Issues:
{issues}

Suggested fixes:
{fixes}

Rewrite the story to address this feedback while:
- Keeping it between 350–600 words
- Preserving the core theme and characters
- Making it even clearer, kinder, and more engaging
- Ending with a short moral that starts with "Moral:"

Return ONLY the improved story, with no explanation or comments.
"""

# User-feedback agent – rewrites the story based on human feedback from the console.
USER_FEEDBACK_PROMPT = """
You are a children's story rewriter.

Here is the current story:
\"\"\"{story}\"\"\"

Here is additional feedback from the human reader:
\"\"\"{feedback}\"\"\"

Rewrite the story to incorporate this feedback while:
- Keeping it appropriate and emotionally safe for ages 5–10
- Keeping roughly the same length
- Keeping a clear beginning, middle, and end
- Ending with a moral starting with "Moral:"

Return ONLY the new story.
"""


def generate_initial_story(user_request: str) -> str:
    """
    Use the STORY_WRITER_PROMPT to generate the first draft
    of a bedtime story based on the user's request.
    """
    prompt = STORY_WRITER_PROMPT.format(user_request=user_request)
    return call_model(prompt, max_tokens=900, temperature=0.8)


def parse_judge_response(raw: str) -> dict:
    """
    Robustly parse the judge's JSON response from raw text.

    The judge is instructed to return JSON only, but LLMs can sometimes
    add extra text. We attempt to locate the first '{' and last '}' and
    parse that substring as JSON.

    On failure, we return a "safe" fallback object with a total_score of 0
    and generic suggestions, so the pipeline can still proceed.
    """
    try:
        start = raw.index("{")
        end = raw.rfind("}")
        json_str = raw[start:end + 1]
        return json.loads(json_str)
    except Exception:
        return {
            "clarity": 0,
            "age_appropriateness": 0,
            "emotional_safety": 0,
            "creativity": 0,
            "narrative_structure": 0,
            "total_score": 0,
            "issues": ["Could not parse judge response as JSON."],
            "suggested_fixes": [
                "Regenerate the story with clearer structure, simpler language, and explicit moral."
            ],
        }


def judge_story(story: str) -> dict:
    """
    Send the story to the judge agent and return its parsed JSON feedback.
    """
    prompt = JUDGE_PROMPT.format(story=story)
    raw = call_model(prompt, max_tokens=600, temperature=0.2)
    return parse_judge_response(raw)


def revise_story(story: str, judge_data: dict) -> str:
    """
    Use the REVISION_PROMPT to improve a story based on judge feedback.

    - Flattens the "issues" and "suggested_fixes" lists into bullet-point text
      so the LLM can easily consume them.
    """
    issues_text = "\n".join(f"- {i}" for i in judge_data.get("issues", []))
    fixes_text = "\n".join(
        f"- {f}" for f in judge_data.get("suggested_fixes", []))

    prompt = REVISION_PROMPT.format(
        story=story,
        issues=issues_text or "- (no issues listed)",
        fixes=fixes_text or "- (no fixes listed)",
    )
    return call_model(prompt, max_tokens=900, temperature=0.7)


def story_pipeline(user_request: str, min_score: int = 40, max_rounds: int = 3, verbose: bool = True):
    """
    Multi-round pipeline that orchestrates:
    1. Initial story generation
    2. Judging
    3. Iterative revisions until score >= min_score or max_rounds is reached.

    Returns:
      - final_story: str
      - final_judge_data: dict
    """
    story = generate_initial_story(user_request)
    judge_data = judge_story(story)

    if verbose:
        print("\n--- Judge evaluation: round 1 ---")
        print("Score:", judge_data.get("total_score"))
        print("Issues:", judge_data.get("issues"), "\n")

    round_idx = 1
    while judge_data.get("total_score", 0) < min_score and round_idx < max_rounds:
        round_idx += 1
        story = revise_story(story, judge_data)
        judge_data = judge_story(story)

        if verbose:
            print(f"--- Judge evaluation: round {round_idx} ---")
            print("Score:", judge_data.get("total_score"))
            print("Issues:", judge_data.get("issues"), "\n")

    return story, judge_data


def main():
    """
    Entry point for the script.

    Flow:
    1) Ask the user what kind of story they want.
    2) Run the automated story pipeline (writer → judge → refiner).
    3) Show the final AI-refined story + judge score/issues.
    4) Enter a loop where the user can keep giving feedback, and for each
       feedback:
         - The story is rewritten by an LLM agent.
         - The updated story is judged again.
         - The new score/issues are shown.
       The loop ends when the user presses Enter with no input.
    """
    print("Welcome to the Bedtime Story Maker (ages 5–10) 🌙\n")

    user_request = input(
        "What kind of story would you like?\n"
        "(For example: 'A dragon who is afraid of the dark but learns to be brave')\n> "
    )

    # Run through the story → judge → refine pipeline
    final_story, final_judge_data = story_pipeline(
        user_request=user_request,
        min_score=40,      # threshold out of 50
        max_rounds=3,      # up to 3 judge/refinement cycles
        verbose=True,
    )

    # Show the first AI-refined story + judge score + judge issues
    print("\n================= YOUR BEDTIME STORY =================\n")
    print(final_story)
    print("\n======================================================")
    print(f"(Internal judge total score: {final_judge_data.get('total_score')})")

    original_issues = final_judge_data.get("issues", [])
    if original_issues:
        print("Internal judge issues for this story:")
        print("- " + "\n- ".join(original_issues))
    else:
        print("Internal judge issues for this story: (none reported)")
    print()  # blank line

    current_story = final_story
    current_judge_data = final_judge_data

    # Allow the user to keep updating the story until they stop
    while True:
        user_feedback = input(
            "Optional: Would you like to tweak the story again?\n"
            "Type feedback like 'shorter', 'more funny', 'add a talking cat', "
            "or press Enter with no text to finish.\n> "
        ).strip()

        if not user_feedback:
            print("\nNo more changes requested.")
            print("Final internal judge score:", current_judge_data.get("total_score"))
            final_issues = current_judge_data.get("issues", [])
            if final_issues:
                print("Final judge issues:")
                print("- " + "\n- ".join(final_issues))
            else:
                print("Final judge issues: (none reported)")
            print("Goodnight! 🌙")
            break

        # Rewrite story based on user feedback
        feedback_prompt = USER_FEEDBACK_PROMPT.format(
            story=current_story,
            feedback=user_feedback,
        )

        updated_story = call_model(
            feedback_prompt,
            max_tokens=900,
            temperature=0.7,
        )

        # Re-run the judge on the updated story
        updated_judge_data = judge_story(updated_story)

        # Update current story & judge data
        current_story = updated_story
        current_judge_data = updated_judge_data

        # Show updated story + updated judge evaluation
        print("\n========= UPDATED STORY =========\n")
        print(current_story)
        print("\n=================================\n")

        print("--- Judge evaluation for updated story ---")
        print("Total score:", updated_judge_data.get("total_score"))
        issues = updated_judge_data.get("issues", [])
        if issues:
            print("Issues:")
            print("- " + "\n- ".join(issues))
        else:
            print("Issues: (none reported)")
        print()

if __name__ == "__main__":
    main()
