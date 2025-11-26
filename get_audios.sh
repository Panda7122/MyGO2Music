mkdir audio -p
yt-dlp \
    -x \
    --audio-format mp3 \
    -o "audio/%(title)s.%(ext)s" "https://youtube.com/playlist?list=PLkauLUNxkz_0yjC7eNkJ__D76atDOtoEH&si=K7KuDfDcV3_wEZtI"