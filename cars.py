# This stores track's definitions

def get_car_params(car_name):
    # Setting paramters for given car name
    # adding new track:
    # always add gg_map for a car
    if car_name == 'Touring':
       gg_map = 'GG1.bmp'

    elif car_name == 'Gokart':
       gg_map = 'GG1_gokart.bmp'

    elif car_name == 'Rally':
       gg_map = 'GG_rally.bmp'

    elif car_name == 'RombusGG':
       gg_map = 'GG_45square.bmp'

    elif car_name == 'SquareGG':
       gg_map = 'GG_square.bmp'

    else:
        raise ValueError('With this name no car is found')

    return gg_map
