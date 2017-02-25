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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ $i
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ $j
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$k"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$k"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ $k
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/"$l"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/en_"$l"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/5s5t_"$i"_"$j"_"$k"_"$l"_en/ $l
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
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ en
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ $i
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ en
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ $j
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/"$k"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ en
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/en_"$k"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s4t_en_"$i"_"$j"_"$k"/ $k
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
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/ en
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/ $i
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/ en
#bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s3t_"$i"_"$j"_en/ $j
#fi
#done
#done

for x in `seq 1 4`
do
i=$langs[$x]
echo $i en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/ en
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s2t_"$i"_en/ $i
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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/ $i
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/ $j
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$k"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/ $k
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/en_"$l"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$l"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s4t_en_"$i"_"$j"_"$k"_"$l"/ $l
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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$k"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/"$l"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$l"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/4s1t_"$i"_"$j"_"$k"_"$l"_en/ None
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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/ $i
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/ $j
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/en_"$k"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$k"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s3t_en_"$i"_"$j"_"$k"/ $k
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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/"$k"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$k"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/3s1t_"$i"_"$j"_"$k"_en/ None
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
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/"$i"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$i"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/ None
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/"$j"_en_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_"$j"_en/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/2s1t_"$i"_"$j"_en/ None
done
done

for x in `seq 1 4`
do
i=$langs[$x]
next=`echo $x+1 | bc`
for y in `seq $next 4`
do
j=$langs[$y]
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/en_"$i"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$i"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/ $i
bash softwares-and-scripts/NNProjects/knmt/nmt_chainer/mr_gxp.sh /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/en_"$j"_test_beam_12 /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/mlnmt.basic.train.config /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/mlnmt.basic.model.best.npz /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.src.raw /windroot/raj/corpora_downloads/IWSLT2016/1s1t_en_"$j"/mlnmt.test.tgt.raw /windroot/raj/corpora_downloads/IWSLT2016/1s2t_en_"$i"_"$j"/ $j
done
done