def discourse_ext(description) -> str:
    question = description.lower().replace("\n", " ").replace("\r", "")
    question = question.replace("[discourse post]", "")
    return question


def offline_msg_ext(description) -> str:
    question = description.partition("Offline transcript:")[2]
    return question


def email_msg_ext(description) -> str:
    # its already clean but returning this for consistency
    return description
