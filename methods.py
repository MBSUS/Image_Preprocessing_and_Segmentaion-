import json
import numpy as np
import os
import time
import cv2

def load_video_as_frames(filename):
    frames=[]
    
    video = cv2.VideoCapture(filename)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    index = 1
    thr = 90
    num_section = 10
    len_section = np.floor(size[0]/ num_section).astype(int)
    success = True

    while success:
        
        success, frame = video.read()
        if not success:
            break
        index += 1
        frames.append(frame[:, :, 0])

    video.release()
    cv2.destroyAllWindows()
    
    return frames,fps,size
    
def show_frames(video,fps=30):
    delay = int(1000 / fps)
    for img in video:
        cv2.imshow("new video", img)
        if cv2.waitKey(delay) == 27:
            break
    cv2.destroyAllWindows()
    return True

def empiric_confidence(x):
    # x,column_id
    confidence_list=[0.7,0.9,1,1,1,1,1,1,0.9,0.7]
    return confidence_list[x]



def init_points(video,size=None,num_section=10,thr=90):
    
    if not size:
        size=[video[0].shape[1],video[0].shape[0]]
    
    len_section = np.floor(size[0]/ num_section).astype(int)
    
    points_video = []
    
    for frame_id in range(len(video)):
        wst, frame = cv2.threshold(video[frame_id], thr, 255, 0)
        points_frame = []
        for section_id in range(num_section):
            tmp=frame[:,section_id*len_section:(section_id+1)*len_section]
            total = sum([sum(i) for i in tmp])
            count=0
            for idx,lice in enumerate(tmp):
                count+=sum(lice)
                if count>=total/2:
                    if True:
                        points_frame.append([(section_id+0.5)*len_section,idx,empiric_confidence(section_id),frame_id])
                    break
        points_video.append(points_frame.copy())
        
    return points_video


def save_video(filename,frames,fps,size):
    
    videoWriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    
    for frame in frames:
        videoWriter.write(frame)
        
    videoWriter.release()

    
    
def load_templates(filepath="template/LAS/ribs/"):
    templates = []
    
    for filename in os.listdir(filepath):
        org = cv2.imread(filepath+filename,0)
        templates.append(org.copy())
        
    return templates

def template_matching(img,templates,thr=0.8):
    
    h,w = [template.shape[0] for template in templates], [template.shape[1] for template in templates]
    
    res_list=[]

    VL=[]
    for i in range(len(templates)):
        res = cv2.matchTemplate(img, templates[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        VL.append([max_val,max_loc,i])
    VL.sort(key=lambda x:-x[0])

    for i in range(3):
        if VL[i][0]>thr:
            top_left = VL[i][1]
            bottom_right = (top_left[0] + w[VL[i][2]], top_left[1] + h[VL[i][2]])
            res_list.append([top_left,bottom_right,VL[i][2],VL[i][0]])
    
    return  res_list      #[coordinates(top_left,bottom_right),template_id,NCC_value(=confidence_value)]

def draw_templates(frame,info,thr=0):
    img=frame.copy()
    for temp in info:
        top_left,bottom_right,temp_id,confidence,frame_id=temp
        if confidence>thr:
            cv2.rectangle(img, top_left, bottom_right, 255, 1)
            cv2.putText(img,"template:"+str(temp_id)+"  Confidence:"+str(round(confidence,2)),(top_left[0],top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1)
    return img

def get_video_info(video,templates):
    video_info=[]
    
    for i in range(len(video)):
        frame_infos = template_matching(video[i],templates)
        L=[]
        if frame_infos:
            for frame_info in frame_infos:
                top_left,bottom_right,temp_id,confidence=frame_info
                L.append([top_left,bottom_right,temp_id,confidence,i])
        if L:
            video_info.append(L.copy())
    return video_info


def median_bottom(info):
    Bottom=[]
    for frame_info in info:
        for piece in frame_info:
            top_left,bottom_right,temp_id,confidence,frame_id=piece
            Bottom.append(bottom_right)
    y_info=[i[1] for i in Bottom]
    y_median = np.median(y_info).astype(int)
    return y_median


def points_error_correct(points_video,median_bottom,scale=None):

    mD = median_D(points_video,median_bottom)
    if scale:
        mD = scale*mD
    
    points_correct = points_video.copy()
    
    for frame_id in range(len(points_correct)):
        for point_id in range(len(points_correct[frame_id])):
            D=np.abs(points_correct[frame_id][point_id][1]-median_bottom)
            if D>mD:
                points_correct[frame_id][point_id][2]=1.1
    return points_correct


def median_bottom(info):
    Bottom=[]
    for frame_info in info:
        for piece in frame_info:
            top_left,bottom_right,temp_id,confidence,frame_id=piece
            Bottom.append(bottom_right[1])
    y_median = np.median(Bottom).astype(int)
    return y_median

def median_D(points_video,median):
    data = []
    for points_frame in points_video:
        for point in points_frame:
            data.append(np.abs(point[1]-median))
    med = np.median(data).astype(int)
    return med

def put_points(video,points_video,fps,median_bottom):
    
    imgs=[cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) for img in video]
    ##imgs[img_id][point_id] : [x,y,confidence,frame_id]
    for img_id in range(len(imgs)):
        for x in range(10):
            if points_video[img_id][x][2]>1:
                color = (0,0,255) 
                cv2.putText(imgs[img_id], 'o', (int(points_video[img_id][x][0]),median_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            color = (0,255,0)
            cv2.putText(imgs[img_id], 'o', (int(points_video[img_id][x][0]),int(points_video[img_id][x][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        cv2.line(imgs[img_id], (0,median_bottom),(imgs[img_id].shape[1],median_bottom),(0, 255, 255))
    
    return imgs