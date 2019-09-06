mode=$1
model=$2
if [ "$mode"x == "train"x ]; then
	echo $mode $model
	sh ./scripts/train/$model.sh
elif [ "$mode"x == "eval"x ]; then
	echo "eval is not implenmented now, refer to README.md"
else
	echo "Not implemented mode" $mode
fi
