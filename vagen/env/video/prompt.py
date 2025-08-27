def system_prompt(**kwargs):
    return """You are an expert in analyzing videos. You will be given a video and a question.

Goal: Answer the question correctly with no more than {max_turns} turns.

Actions you can take: 
 In each turn, you can choose to retrieve more frames from the video or provide your answer. But you need to answer the question when you reach limit of retrieval times.
1、retrieve (provide specific frame index range). 
2、answer (final response to the question)

Analysis Process:
You must conduct reasoning inside <think> and </think> first every time you get new frames. After reasoning, if you lack some information of the video to answer the question, you can send a retrive request by <retrive> start_frame, end_frame </retrive> and you can get sampled frames between this range in the next turn. \
If you have enough information, you can directly provide the answer inside <answer> and </answer>.

Note: Each observation will show the current turn number (e.g., "Turn 1/3") to help you track your progress.

‌Key Constraints‌:
1、 ‌Answer the question in no more than {max_turns} turns.
2、 Send retrive request ONLY if critical details are missing.
"""

def system_prompt_v2(**kwargs):
    return """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. You are an assistant in analyzing videos. Your will be given a video and a question.  Goal: Answer the question correctly with no more than {max_turns} turns.
"""


def free_think_format_prompt_v2(add_example=False, **kwargs):
    max_frame_idx = kwargs.get("max_frame_idx", "N/A")
    problem_type = kwargs.get("problem_type", None)
    assert problem_type is not None, "problem_type is None"
    turn_num = kwargs.get("turn_num", 1)
    max_turns = kwargs.get("max_turns", "N/A")
    
    # Get type-specific instruction from TYPE_TEMPLATE
    answer_type_instruction = TYPE_TEMPLATE.get(problem_type, "")
    
    intermediate_turn_prompt = f"""Format Template: 
<think>...</think><answer>...</answer> or <think>...</think><retrieve>...</retrieve> \
Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process.Provide your detailed reasoning between the <think> and </think> tags. 
If you have enough information, {answer_type_instruction}.
If you lack some information, think about the most relevant frame index range of the information you need, then you can retrieve dense frames in the range by sending a retrive request by <retrive> start_frame, end_frame </retrive>. IMPORTANT: start_frame and end_frame must be integers smaller than {max_frame_idx}.

"""
    final_turn_prompt = f"""Format Template: 
    <think>...</think><answer>...</answer> \
   Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags.  {answer_type_instruction}
    """

    if turn_num >= max_turns:
        base_prompt = final_turn_prompt
    else:
        base_prompt = intermediate_turn_prompt

    return base_prompt


def free_think_format_prompt(add_example=True, **kwargs):
    max_frame_idx = kwargs.get("max_frame_idx", "N/A")
    problem_type = kwargs.get("problem_type", None)
    assert problem_type is not None, "problem_type is None"
    turn_num = kwargs.get("turn_num", 1)
    max_turns = kwargs.get("max_turns", "N/A")
    
    # Get type-specific instruction from TYPE_TEMPLATE
    type_instruction = TYPE_TEMPLATE.get(problem_type, "")
    

    intermediate_turn_prompt = f"""Format Template: 
<think>...</think><answer>...</answer> or <think>...</think><retrieve>...</retrieve> \

If you have enough information, first think about your answer, then you can provide the answer inside <answer> and </answer>.{type_instruction}.
If you lack some information, think about what information you need, then you can send a retrive request by <retrive> start_frame, end_frame </retrive>. IMPORTANT: start_frame and end_frame must be integers smaller than {max_frame_idx}].

"""
    final_turn_prompt = f"""Format Template: 
    <think>...</think><answer>...</answer>
    You must provide your final answer now.
    """

    if turn_num >= max_turns:
        base_prompt = final_turn_prompt
    else:
        base_prompt = intermediate_turn_prompt


    if add_example:
        retrieve_example = f"""e.g. <think>It seems the girls are playing at the begining of the video. But I need to check what kind of sports they are playing.</think><retrieve>5,10</retrieve>"""
        # Customize example based on problem type
        if problem_type == "multiple choice":
            answer_example = f"""e.g. <think>I am sure that the girls are playing football. I would choose B.</think><answer>B</answer>"""
        elif problem_type == "free-form":
            answer_example = f"""e.g. <think>I think the car is turning left.</think><answer>Turn left.</answer>"""
        else:
            assert False, f"problem_type:{problem_type} not supported"

        if turn_num < max_turns:
            return base_prompt + '\n' + answer_example + '\n' + retrieve_example
        else:
            return base_prompt + '\n' + answer_example
    return base_prompt


def video_r1_format_user_prompt(**kwargs):
    observation = kwargs.get("observation", "")
    frame_idx_list = kwargs.get("frame_idx_list", [])

    problem_type = kwargs.get("problem_type", None)
    problem = kwargs.get("problem", "")
    options = kwargs.get("options", [])

    prompt = "Here are some sample frames from the video: " 
    prompt += "\n".join([f"{observation}" for idx in frame_idx_list])

    
    if problem_type == "multiple choice":
        question = problem + " Options:\n"
        for op in options:
            question += op + "\n"
    else:
        question = problem

    prompt += "\n" + QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[problem_type]

    return prompt

    

def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "")
    problem_type = kwargs.get("problem_type", None)
    assert problem_type is not None, "problem_type is None"
    problem = kwargs.get("problem", "")
    options = kwargs.get("options", [])
    frame_idx_list = kwargs.get("frame_idx_list", [])
    max_frame_idx = kwargs.get("max_frame_idx", "N/A")
    turn_num = kwargs.get("turn_num", 1)
    max_turns = kwargs.get("max_turns", "N/A")

    #print(f"[Debug] init_observation_template: turn_num={turn_num}, max_turns={max_turns}")
    if problem_type == "multiple choice":
        problem = f"{problem}\nOptions: {options}"

    if frame_idx_list is not None and len(frame_idx_list) > 0:
        frames_str = "\n".join([f"frame_idx:{idx}, {observation}" for idx in frame_idx_list])
    else:
        frames_str = observation

    turn_info = f"Turn {turn_num}"


    if max_turns != "N/A" and turn_num >= max_turns:
        action_prompt = "You have reached the maximum number of turns. You must provide your final answer now."
    else:        
        action_prompt = f"You can choose to retrieve more frames or provide your answer."


    return f"""
{turn_info}
Now you are given {len(frame_idx_list)} selected frames from the video, with frame_idx_list: {frame_idx_list} 

Frames:
{frames_str}
Answer the following problem based on the frames:
{problem}
{action_prompt}
"""


def invalid_state_observation_template(**kwargs):
    problem_type = kwargs.get("problem_type", None)
    assert problem_type is not None, "problem_type is None"
    problem = kwargs.get("problem", "")
    options = kwargs.get("options", [])
    if problem_type == "multiple choice":
        problem = f"{problem}\nOptions: {options}"

    turn_num = kwargs.get("turn_num", 1)
    max_turns = kwargs.get("max_turns", "N/A")
    turn_info = f"Turn {turn_num}"

    if max_turns != "N/A" and turn_num >= max_turns:
        action_prompt = "You have reached the maximum number of turns. You must provide your final answer now."
    else:        
        action_prompt = f"You can choose to retrieve more frames or provide your answer."

    return f"""
{turn_info}
Now you have to observe the given frames in the previous turn.

Answer the following problem based on the previous frames:
{problem}
{action_prompt}
"""

def action_template(**kwargs):
#     valid_action, observation= kwargs.get("valid_action", ""), kwargs.get("observation", "The player is on the above the target")
#     return f"""After your answer, the extracted valid action is {valid_action}.
# After that, the observation is:
# {observation}
# Decide your next action(s).
# """
    return ""

# Search-R1
# def make_prefix(dp, template_type):
#     question = dp['question']

#     # NOTE: also need to change reward_score/countdown.py
#     if template_type == 'base':
#         """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
#     else:
#         raise NotImplementedError
#     return prefix

# Video-R1/src
SYSTEM_PROMPT_VIDEO_R1_EVAL = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
        )

QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags. Conclude your answer in not more than 20 words.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}



# Dictionary mapping format names to their corresponding functions
format_prompt = {
    "free_think": free_think_format_prompt,
    "free_think_v2": free_think_format_prompt_v2,
    #"video_r1": video_r1_format_prompt,
    #"no_think": no_think_format_prompt,
}


# parsing funcs
import re
from typing import Dict, List
import json

def extract_option_letters(options: List[str]) -> List[str]:
    """
    Extract valid option letters from options list.
    
    Args:
        options: List of option strings like ["A. To use...", "B. To clean...", ...]
        
    Returns:
        List of valid option letters like ["A", "B", "C", "D"]
    """
    valid_letters = []
    for option in options:
        if option and len(option) > 0:
            # Extract the first character and check if it's a letter
            first_char = option.strip()[0].upper()
            if first_char.isalpha():
                valid_letters.append(first_char)
    return valid_letters

def check_retrieve_format(response: str) -> bool:
    """
    Check if the retrieve response is in the format of:start_idx, end_idx, eg. 2,8
    1、start_frame and end_frame are integers
    2、separated by a comma
    3、start_frame is less than end_frame
    4、start_frame and end_frame are in the range of the video

    """
    pattern = r'^\s*(\d+)\s*,\s*(\d+)\s*$' # NOTE: start_idx and end_idx are integers
    match = re.match(pattern, response.strip(), re.DOTALL)
    if match is not None:
        # NOTE: start_idx and end_idx are integers
        # Simply convert to int, no need to check if they are integers
        start_idx = int(match.group(1))
        end_idx = int(match.group(2))
        return start_idx <= end_idx and start_idx >= 0, start_idx, end_idx
    return False, None, None

def check_retrieve_valid(start_idx: int, end_idx: int, max_frame_idx: int) -> bool:
    """
    Check if the retrieve is valid
    1、start_idx is not greater than end_idx
    2、start_idx and end_idx are in the range of the video
    """
    return (start_idx >= 0 and 
            end_idx <= max_frame_idx and 
            start_idx <= end_idx)


def parse_videnv_llm_raw_response(response: str, max_frame_idx: int, video_dict=None, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think>...</think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <answer> tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> or <retrieve> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    - action_valid: whether the action is valid
    - action_type: type of action ("answer" or "retrieve")
    - error_type: type of error if any ("none", "format_error", "retrieve_format_error", "retrieve_range_error")
    - error_message: detailed error message
    """
    #response = response.replace("<image>","") # ??
    # Get the problem type from video_dict
    problem_type = video_dict.get("problem_type", None)
    assert problem_type is not None, "problem_type is None"

    # Initialize error tracking
    error_type = "none"
    error_message = ""

    #Pattern to check for content strictly in the format <think>...</think><answer>...</answer>
    strict_pattern_a = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match_a = re.match(strict_pattern_a, response.strip(), re.DOTALL)

    strict_pattern_r = r'^\s*<think>(.*?)</think>\s*<retrieve>(.*?)</retrieve>\s*$'
    strict_match_r = re.match(strict_pattern_r, response.strip(), re.DOTALL)

    #print(f"[Debug]strict_match_r: {strict_match_r}")
    # Pattern to extract content from think and answer tags
    if strict_match_a is not None:
        extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        action_type = "answer"
    elif strict_match_r is not None:
        extraction_pattern = r'<think>(.*?)</think>\s*<retrieve>(.*?)</retrieve>'
        action_type = "retrieve"
    else:
        action_type = None
        error_type = "format_error"
        error_message = "Not follow the expected format: <think>...</think><answer>...</answer> or <think>...</think><retrieve>...</retrieve>"
        return {
            "llm_raw_response": response,
            "llm_response": "",
            "think_content": "",
            "action_content": "",
            "action_type": action_type,
            "action_valid": False,
            "format_correct": False,
            "error_type": error_type,
            "error_message": error_message
        }

    match = re.search(extraction_pattern, response, re.DOTALL)

    # format reward
    format_correct = strict_match_a is not None or strict_match_r is not None

    # retrieve format need furthercheck
    if action_type == "retrieve":
        format_correct_inner, start_idx, end_idx = check_retrieve_format(match.group(2))
        if not format_correct_inner:
            error_type = "retrieve_format_error"
            error_message = f"Retrieve format is incorrect. Expected format: 'start_frame,end_frame' (e.g., '10,30'), got: '{match.group(2).strip()}'"
        format_correct = format_correct and format_correct_inner
        if format_correct:
            action_valid = check_retrieve_valid(start_idx, end_idx, max_frame_idx)
            if not action_valid:
                error_type = "retrieve_range_error"
                error_message = f"Retrieve range is invalid. start_idx={start_idx}, end_idx={end_idx}, max_frame_idx={max_frame_idx}. Requirements: 0 <= start_idx <= end_idx <= max_frame_idx"
        else:
            action_valid = False
    elif action_type == "answer":
        if problem_type == "multiple choice":
            # Extract valid option letters from options (e.g., "A", "B", "C", "D")
            options = video_dict.get("options", [])
            valid_letters = extract_option_letters(options)
            
            # Check if the answer letter is in valid options
            answer_letter = match.group(2).strip().upper()
            action_valid = answer_letter in valid_letters
            
        elif problem_type == "free-form":
            # check if the answer not too long
            answer_length = len(match.group(2).strip())
            length_limit = 50
            if answer_length > length_limit:
                action_valid = False
                error_message = f"Answer is too long. Length limit: {length_limit}, got: {answer_length}"
                print(f"[Debug] {error_message}")
            else:
                action_valid = True
        elif problem_type == "numerical":
            # check if the answer is a number
            try:
                float(match.group(2).strip())
                action_valid = True
            except:
                action_valid = False
                error_message = f"Answer is not a number. Got: {match.group(2).strip()}"
        else:
            assert False, f"problem_type:{problem_type} not supported"
    else :
        action_valid = True  # answer actions are always valid if format is correct

    if not format_correct:
        think_content, action_content = "", ""
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

    llm_response = "<think>" + think_content.strip() + "</think>" + f"<{action_type}>" + action_content.strip() + f"</{action_type}>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "action_type": action_type,
        "action_valid": action_valid,
        "format_correct": format_correct,
        "error_type": error_type,
        "error_message": error_message
    }

