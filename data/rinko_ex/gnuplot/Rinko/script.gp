#load 'script.gp'で実行

#絶対パスへ移動
cd '/home/mashimo/program/Cepstrum/0506Experiment/gnuplot'

#凡例の位置(center, left, right, outside, top, bottom等)
set key top left

#xy軸のラベル・範囲設定
#set xl "hoge"でもOK
set xlabel "True Blur Length"
set ylabel "Estimated Blur Length"
#set title "title hogefuga"
#set yrange[-20:20]

# プロットデータからグラフ生成
plot "./result_ex_blur.dat" using 1:2 with l title "Ground Truth"
replot "./result_ex_blur.dat" using 1:3 with p title "Blur Length Estimation"

#eps画像生成
set terminal postscript eps color "Arial"
#出力先のファイル指定(相対パス)
set output "result_ex_blur.eps"

#描画
replot
