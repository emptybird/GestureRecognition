from gtts import *

list = [str(i) for i in range(0,6)]

for i in list:
    filename = 'sound/' + i + '.mp3'
    tts = gTTS(i, lang='zh-cn')
    tts.save(filename)