import numpy as np
def text_to_labels(text):
    labels=[]
    temp=text.lower()
    
    for i in range(len(temp)):
        
        if temp[i]>='a' and temp[i]<='z':
            
            if(i!=0 and temp[i]==temp[i-1]):
                labels.append(27)
                labels.append(ord(temp[i])-ord('a'))
                
            else :labels.append(ord(temp[i])-ord('a'))
                
        elif temp[i]==' ':labels.append(26)
    return labels

def text_to_labels_original(text):
    ret = []
    temp=text.lower()

    for char in temp:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char >='0' and char<='9':
            ret.append(ord(char) - ord('0')+26)
        elif char == '?': ret.append(36)
        elif char == ',': ret.append(37)
        elif char == '.': ret.append(38)
        elif char == '!': ret.append(39)
        elif char == ':': ret.append(40)
        elif char == ' ': ret.append(41)
    return ret

def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

def pad(video, length):

        video_length=len(video)

        pad_length = max(length - video_length, 0)
        video_length = min(length, video_length)

        mouth_padding = np.ones((pad_length, video.shape[1], video.shape[2], video.shape[3]), dtype=np.float32) * 0

        video_mouth = np.concatenate((video[0:video_length], mouth_padding), 0)

        return video_mouth

