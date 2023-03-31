import youtube_dl

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
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

if __name__ == "__main__":
    downloadYTmp3("https://www.youtube.com/watch?v=fRed0Xmc2Wg","output.wav")