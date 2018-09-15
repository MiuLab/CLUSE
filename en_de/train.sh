if [ ! -d $1 ]; then
	  mkdir $1
fi
python2.7 main.py --dataset_dir ../data/en_de/ --log_path $1.txt --save_dir $1/ --major_weight $2 --reg_weight $3
