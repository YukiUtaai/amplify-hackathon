# amplify-hackathon

## 実行方法

1. ファイル *kyopro_arc001B_amplify.py* の、`client.token`のコメントアウトを外してアクセストークンを入力する

2. 初期温度t1と目標温度t2を設定する
（場合によっては最大操作回数Nを変更する。リモコンの機能を変更して遊ぶこともできます）
```shell
t1, t2 = -39, 38 # 初期温度、目標温度
T = t2 - t1 # 初期温度と目標温度の差→初期温度と目標温度に依らず、この値が重要となる
N = 12 # リモコンを最大で何回操作するか→Tを大きくする場合は大きくする必要がある
R = [0, -1, 1, -5, 5, -10, 10] # リモコンの機能
```

3. *kyopro_arc001B_amplify.py* を実行する
```shell
$ python sudoku_sample_amplify.py
```

## 実行結果

```
結果（0℃変更含む）
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 0℃変更
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -1℃変更
[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] 1℃変更
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -5℃変更
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 5℃変更
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -10℃変更
[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1] 10℃変更
--------------------------
結果（0℃変更含まない）
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -1℃変更
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0] 1℃変更
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -5℃変更
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 5℃変更
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -10℃変更
[1, 1, 0, 1, 0, 1, 0, 1, 1, 1] 10℃変更
--------------------------
<Figure size 640x480 with 1 Axes>
操作回数 10
```

下記では<Figure size 640x480 with 1 Axes>となっていますが、結果の画像が表示されます。

・jupyter notebookで実行する場合は2回実行する必要があるかもしれません）

・WSL環境だとグラフが表示されない場合があります。下記のようにすることで画像を出力できます。

```
# plt.show()
plt.savefig("img.png")
```



## 提出前チェック


- [x] README.mdの手順通りにして、プログラムが実行できる
- [x] 説明用スライドを用意した 
- [x] アクセストークンはリポジトリに含まれていない
- [x] MIT Licenseにした
