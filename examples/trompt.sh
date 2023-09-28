for dataset in bank-marketing compas-two-years MagicTelescope
do
echo $dataset
python trompt.py --dataset $dataset
done

# electricity eye_movements california credit jannis pol 