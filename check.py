import pandas as pd

anno_df = pd.read_csv('data/train_with_labels.csv')
labels_df = pd.read_csv('data/train_annotations.csv')
new_df = pd.merge(anno_df, labels_df, on='image_id')
print(new_df.columns)
new_df = new_df[['image_id', 'width', 'height', 'flickr_url', 'coco_url', 'label_count', 'file_name', 'category_id', 'area', 'bbox']]
# new_df.to_csv('./data/train_data.csv', index=False)
print(new_df.info())