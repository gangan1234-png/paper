import requests
import pandas as pd

key = '百度地图key'
coordtype = 'wgs84ll'

df = pd.read_csv('Set of regional coordinates.csv',sep=',')
print(df.head())

for i in range (len(df)):
      id = df.iloc[i,0]
      x = df.iloc[i,1]
      y = df.iloc[i, 2]
      save_path = 'pic/'+str(id) + '.png'
      # x = '116.497906'
      # y = '39.793173'
      url = 'https://api.map.baidu.com/panorama/v2?' \
            'ak='+key+'&width=1024&height=512&coordtype='+coordtype+'' \
            '&location='+str(x)+','+str(y)+'&fov=360'
      print(url)
      try:
            data = requests.get(url)
            open(save_path,'wb').write(data.content)
      except:
            pass