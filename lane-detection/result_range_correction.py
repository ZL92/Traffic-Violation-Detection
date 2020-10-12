import json
import os
import copy

path = '/home/zl92/Desktop/result'

files = os.listdir(path)
for file in files:
    new_results_json = []
    print(path + '/' + file)
    f = open(path + '/' + file, 'r')
    f = json.load(f)
    if f == []:
        continue
    else:
        for idx, result in enumerate(f):
            print('idx: ', idx)
            print(result)
            new_result = copy.deepcopy(result)
            new_result['xs'] = int(result['xs'] * 1920 / 1280)
            new_result['xe'] = int(result['xe'] * 1920 / 1280)
            new_result['ws'] = int(result['ws'] * 1920 / 1280)
            new_result['we'] = int(result['we'] * 1920 / 1280)

            new_result['ys'] = int(result['ys'] * 1080 / 720)
            new_result['ye'] = int(result['ye'] * 1080 / 720)
            new_result['hs'] = int(result['hs'] * 1080/720)
            new_result['he'] = int(result['he'] * 1080/720)

            new_results_json.append(new_result)
    with open('./result/' + file, 'w', encoding='utf-8') as push_file:
        json.dump(new_results_json, push_file, ensure_ascii=False)
