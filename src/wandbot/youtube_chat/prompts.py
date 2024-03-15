summaries_prompt_template = """You are a proficient AI with a specialty in comprehension and summarization.

Generate increasingly concise, entity-dense summaries of the provided video transcript. 

Repeat the following 2 steps 5 times. 

Step 1. Identify 1-3 informative entities (";" delimited) from the video which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 

A missing entity is:
- relevant to the main story, 
- specific yet concise (5 words or fewer), 
- novel (not in the previous summary), 
- faithful (present in the video transcript), 
- anywhere (can be located anywhere in the video transcript).

Guidelines:

- The first summary should be long (7-10 sentences, ~120 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this transcript discusses") to reach ~120 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the video discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the video transcript. 
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 

Remember, use the exact same number of words for each summary.
"""


keypoints_prompt_template = (
    "You are a proficient AI with a specialty in distilling information into key points. "
    "Based on the following video transcript, identify and list the main points that were discussed or brought up. "
    "These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. "
    "Your goal is to provide a list that someone could read to quickly understand what was talked about."
)
