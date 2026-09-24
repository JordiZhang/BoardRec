import pandas as pd
import requests
import configparser
import time
import xml.etree.ElementTree as ET

config = configparser.ConfigParser()
config.read('../utils/config.ini')
api = config['api']

def get_user_collection(user):
    # own = 1, prevowned = 1, want = 1, wishlist = 1, wishlist_priority = 1.
    url = 'https://boardgamegeek.com/xmlapi2/collection?username=' + user + '&excludesubtype=boardgameexpansion&fortrade=0&wanttoplay=0&wanttobuy=0&preordered=0&stats=1'
    fail = False
    i = 0

    # authorization
    header = {"Authorization": f"Bearer {api['auth_token']}"}

    # BGG API call
    while True:
        i += 1
        time1 = time.time()
        response = requests.get(url, headers=header)


        # added delay between calls due to rate throttling
        time2 = time.time()
        diff = time2 - time1
        delay = max(0, 5-diff)
        time.sleep(delay)

        # break if successful
        if response.status_code == 200:
            break

        # fail API call if takes too many attempts
        if i >= 5:
            fail = True
            break

    # check if API call failed
    if fail:
        raise Exception('API call fail, API down or user does not exist.')

    response = response.content.decode('utf-8')

    root = ET.fromstring(response)

    users = []
    games = []
    for item in root:
        gamename = item.find('name').text
        users.append(user)
        games.append(gamename)

    collection = {'username': users, 'name': games}
    collection = pd.DataFrame(collection)
    return collection
