def discourseExt(description) -> str:
    question = description.lower().replace('\n', ' ').replace('\r', '')
    question = question.replace('[discourse post]','')
    question = question[:4095]
    return question

def offlineMessageExt(description) -> str:
    question = description.partition("Offline transcript:")[2]
    return question

def emailMsgExt(description) -> str:
    # its already clean but returning this for consistency
    return description