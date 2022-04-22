import sys
import csv
import pickle
import os

admnote_folder = sys.argv[1]

note_texts = {}
for file in os.listdir(admnote_folder):
    reader = csv.reader(open(os.path.join(admnote_folder, file)))
    next(reader, None)
    for row in reader:
        note_texts[int(row[0])] = row[1]

pmv_labels = pickle.load(open('pmv_labels.pkl', 'rb'))
if not os.path.isdir('mechanical_ventilation'):
    os.mkdir('mechanical_ventilation')
train_file = open('mechanical_ventilation/pmv_train.csv', 'w')
dev_file = open('mechanical_ventilation/pmv_dev.csv', 'w')
test_file = open('mechanical_ventilation/pmv_test.csv', 'w')
train_writer = csv.writer(train_file)
dev_writer = csv.writer(dev_file)
test_writer = csv.writer(test_file)
train_writer.writerow(['id', 'text', 'label'])
dev_writer.writerow(['id', 'text', 'label'])
test_writer.writerow(['id', 'text', 'label'])

for note in pmv_labels:
    if pmv_labels[note][-1] == 'train':
        train_writer.writerow([note, note_texts[note], pmv_labels[note][0]])
    if pmv_labels[note][-1] == 'val':
        dev_writer.writerow([note, note_texts[note], pmv_labels[note][0]])
    if pmv_labels[note][-1] == 'test':
        test_writer.writerow([note, note_texts[note], pmv_labels[note][0]])

train_file.close()
dev_file.close()
test_file.close()
