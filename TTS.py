from gtts import *
list = ['0','1','2','3','4','5']

for i in list:
    filename = 'sound/' + i + '.mp3'
    tts = gTTS(i, lang='zh-cn')
    tts.save(filename)