import argparse
import json
import pickle
import glob
import cv2
from tqdm import tqdm
import config as args

def prepare(prepared_annotations, folder,count):
    images=glob.glob(folder+"/*.jpg")
    json_files=glob.glob(folder+"/*.json")
    files=list(zip(images,json_files))
    
    for i,j in tqdm(files):
        # print(j)
        assert i.split(".jpg")[0]==j.split(".json")[0]
        annot={}
        img=cv2.imread(i)
        width,height,_=img.shape
        assert width==height==368
        with open(j, 'r') as f:
            data = json.load(f)
        annot['keypoints']=data['hand_pts']
        annot['img_width']=width
        annot['img_height']=height
        annot['img_paths']=i
        annot['image_id']=count
        count+=1
        prepared_annotations.append(annot)
    return prepared_annotations, count

if __name__ == '__main__':
    prepared_annotations=[]
    count=0
    for i in args.datasets:
        prepared_annotations,count=prepare(prepared_annotations,i,count=0)

    with open(args.output_pkl_file, 'wb') as f:
        pickle.dump(prepared_annotations, f)

