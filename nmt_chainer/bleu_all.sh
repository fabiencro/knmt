langs=(ar cs de fr)

for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
nextnext=`echo $y+1 | bc`
for z in `seq $nextnext 4`
do
k=$langs[$z]
nextnextnext=`echo $z+1 | bc`
for a in `seq $nextnextnext 4`
do
l=$langs[$a]
echo $i $j $k $l
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$k"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$k"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$l"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$l"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done
done
done


#for x in `seq 1 4`
#do
#i=$langs[$x]
#next=`echo $x+1 | bc`
#for y in `seq $next 4`
#do
#j=$langs[$y]
#nextnext=`echo $y+1 | bc`
#for z in `seq $nextnext 4`
#do
#k=$langs[$z]
#echo $i $j $k
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$k"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$k"_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw
#done
#done
#done


#for x in `seq 1 4`
#do
#i=$langs[$x]
#next=`echo $x+1 | bc`
#for y in `seq $next 4`
#do
#j=$langs[$y]
#if [[ $i != $j ]]
#then
#echo $i $j
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw
#python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw
#fi
#done
#done

for x in `seq 1 4`
do
i=$langs[$x]
echo $i en
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged
done

for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
nextnext=`echo $y+1 | bc`
for z in `seq $nextnext 4`
do
k=$langs[$z]
nextnextnext=`echo $z+1 | bc`
for a in `seq $nextnextnext 4`
do
l=$langs[$a]
echo $i $j $k $l
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$k"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$l"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done
done
done


for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
nextnext=`echo $y+1 | bc`
for z in `seq $nextnext 4`
do
k=$langs[$z]
nextnextnext=`echo $z+1 | bc`
for a in `seq $nextnextnext 4`
do
l=$langs[$a]
echo $i $j $k $l
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$k"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$l"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done
done
done


for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
nextnext=`echo $y+1 | bc`
for z in `seq $nextnext 4`
do
k=$langs[$z]
echo $i $j $k
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$k"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done
done



for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
nextnext=`echo $y+1 | bc`
for z in `seq $nextnext 4`
do
k=$langs[$z]
echo $i $j $k
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$k"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done
done


for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
echo $i $j
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/"$i"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/"$j"_en_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done

for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
echo $i $j
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/en_"$i"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/en_"$j"_test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done
done


for x in `seq 1 4`
do
i=$langs[$x]
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/test_beam_12/src-shard.trans.unk_replaced.restored.merged 
python bleu_computer.py /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/test_beam_12/src-shard.trans.unk_replaced.restored.merged 
done