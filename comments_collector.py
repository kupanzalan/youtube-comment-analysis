import requests
import json
from apiclient.discovery import build
from csv import writer
from urllib.parse import urlparse, parse_qs

def build_service(api_key):
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,
                 YOUTUBE_API_VERSION,
                 developerKey=api_key)

def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]

def comments_helper(video_ID, api_key, service, no_of_comments):

    comments = []

    response = service.commentThreads().list(
        part="snippet",
        videoId = video_ID,
        textFormat="plainText").execute()
    page = 0

    while len(comments) < no_of_comments:
        for item in response['items']:
            page += 1
            comment = item["snippet"]["topLevelComment"]
            comment_id = item['snippet']['topLevelComment']['id']
            print('\n')
            print('Comment no. ', page)
            print('comment_id: ', comment_id)
            text = comment["snippet"]["textDisplay"]
            print('text: ', text)
            comments.append(text)
            # commentsId.append(comment_id)

        if 'nextPageToken' in response:

            response = service.commentThreads().list(part="snippet",
                videoId = video_ID,
                textFormat="plainText",
                pageToken=response['nextPageToken']
            ).execute()
        else:
            break

    return comments


