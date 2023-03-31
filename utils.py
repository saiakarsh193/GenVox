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

def secToFormattedTime(seconds):
    seconds = int(seconds)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    days = int(hours / 24)
    seconds -= minutes * 60
    minutes -= hours * 60
    hours -= days * 24
    if (hours > 0):
        return f"{days}-{hours}:{minutes}:{seconds}"
    return f"{hours}:{minutes}:{seconds}"
