#load 'script_forerror.gp'で実行

#絶対パスへ移動
cd '/home/mashimo/program/Cepstrum/0506Experiment/gnuplot'

#凡例の位置(center, left, right, outside, top, bottom等)
set key top left

#xy軸のラベル・範囲設定
#set xl "hoge"でもOK
set xlabel "True Blur Length"
set ylabel "Blur Length Error"
set mytics 10 #軸目盛りを10分割したサブ目盛り表示
#set title "title hogefuga"
set yrange[-30:30]
set xrange[0:300]

# プロットデータからグラフ生成
plot "./result_ex_blur.dat" using 1:4 with lp title "Estimation Error"

#eps画像生成
set terminal postscript eps color "Arial"
#出力先のファイル指定(相対パス)
set output "result_ex_blur_error.eps"

#描画
replot
