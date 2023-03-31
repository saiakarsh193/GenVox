import random
import yt_dlp

def downloadYTmp3(link, target):
    # download options for youtube_dl
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': target,
        'writesubtitles' : target,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

def getRandomHexName(size=60):
    return hex(random.getrandbits(size))[2: ]
