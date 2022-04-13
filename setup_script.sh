DATASET_PATH="./dataset"

declare -a DATAFILES=("Indian_pines_corrected.mat"
                    "Indian_pines_gt.mat"
                    "PaviaU.mat"
                    "PaviaU_gt.mat"
                    "KSC.mat"
                    "KSC_gt.mat"
                    "Salinas_corrected.mat"
		    "Salinas_gt.mat")
declare -a DATAURL=("http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
                    "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
                    "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
                    "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
                    "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat"
                    "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat"
                    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_corrected.mat"
		    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_gt.mat")

if [ ! -d "$DATASET_PATH" ]; then
  # Take action if $DIR exists. #
  echo "Creaating ${DATASET_PATH}..."
  mkdir "$DATASET_PATH"
fi
length=${#DATAFILES[@]}
for (( i = 0; i < length; i++ )); 
do
    if [ -f "$DATASET_PATH/${DATAFILES[i]}" ]; then
        ### Take action if $DIR exists ###
        echo "${DATAFILES[i]} File exists..."
    else
        ###  Control will jump here if $DIR does NOT exists ###
        echo "${DATAFILES[i]} file doesn't exists..."
        wget "${DATAURL[i]}" -P "$DATASET_PATH"
    fi
done
