def count_blacks_and_whites(right_code, user_code):
    # Kopieer de arrays om ze niet buiten de scope van de functie te veranderen:
    right_code = right_code[:]
    user_code = user_code[:]

    # Verwijder eerst alle pinnetjes die goed staan
    num_blacks = 0
    for i in range(len(user_code) - 1, -1, -1):  # Ga van achter naar voor i.v.m. concurrent modification
        if user_code[i] == right_code[i]:
            right_code.pop(i)
            user_code.pop(i)
            num_blacks += 1

    # Deze kopie van de right_code array wordt aangepast tijdens het uitvoeren
    # van de functie en bevat alle kleuren die nog over (left) zijn. (Dit om de telling
    # goed te laten verlopen als in de right_code meerdere pinnetjes van dezelfde kleur zijn.)
    right_pins_left = right_code[:]
    num_whites = 0  # Gaat worden incremented en returned
    for i in range(len(user_code)):
        if user_code[i] in right_pins_left:
            num_whites += 1
            right_pins_left.remove(user_code[i])
    return num_blacks, num_whites
