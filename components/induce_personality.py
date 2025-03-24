import itertools

personality_types = [["extroverted", "introverted"], ["agreeable", "antagonistic"], ["conscientious", "unconscientious"], ["neurotic", "emotionally stable"], ["open to experience", "closed to experience"]]


def construct_big_five_words(persona_type: list):
    """Construct the list of personality traits

    e.g., introverted + antagonistic + conscientious + emotionally stable + open to experience
    """
    options = list(persona_type)
    assert options[0] in ["extroverted", "introverted"], "Invalid personality type"
    assert options[1] in ["agreeable", "antagonistic"], "Invalid personality type"
    assert options[2] in ["conscientious", "unconscientious"], "Invalid personality type"
    assert options[3] in ["neurotic", "emotionally stable"], "Invalid personality type"
    assert options[4] in ["open to experience", "closed to experience"], "Invalid personality type"
    last_item = "and " + options[-1]
    options[-1] = last_item
    return ", ".join(options)

def build_personality_prompt(persona_type: list):
    return "You are a character who is {}.".format(construct_big_five_words(persona_type))



if __name__ == "__main__":
    count = 0
    for persona_type in itertools.product(*personality_types):
        system_prompt = "You are a character who is {}.".format(construct_big_five_words(persona_type))
        print(system_prompt)
        print("\n")
        count += 1
        if count == 5:
            break
