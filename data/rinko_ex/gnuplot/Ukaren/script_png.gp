# load 'script.gp'で実行

# 絶対パスへ移動
cd '/home/mashimo/program/cepstrum/0506Experiment/gnuplot/Ukaren'

# 凡例の位置(center, left, right, outside, top, bottom等)
set key bottom right font "Times New Roman, 20"

# xy軸のラベル・範囲設定
# set xl "hoge"でもOK
set xlabel "True Blur Length(pixel)" font "Times New Roman, 20"
set ylabel "Estimated Blur Length(pixel)" font "Times New Roman, 20"
# set title "title hogefuga"
set yrange[50:452]
set xrange[50:410]

# プロットデータからグラフ生成
# lp=linespoints,lt=linetype，lw=linewidth，pt=pointtype，ps=pointsize
# ex) with lp lt3 lw 2 pt 5 ps 2
plot "./result_ex_blur.dat" using 1:2 with l title "Ground Truth" lw 5
replot "./result_ex_blur.dat" using 1:3 with p title "Blur Length  Estimation" lc rgb "blue" pt 4 ps 7

# eps画像生成
# set terminal postscript eps color enhanced "Times-Roman"
set terminal png
# 画像サイズ指定（デフォルト640x480）
set term png size 920, 640
# 出力先のファイル指定(相対パス)
set output "result_blur.png"

# 描画
replot

# http://www.proton.jp/main/apps/gnuplotadjust.html
# show colorname

