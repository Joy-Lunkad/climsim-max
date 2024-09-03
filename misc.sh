pip install kaggle
kaggle -v
gsutil cp gs://us2-climsim/kaggle.json ./
cp kaggle.json ~/.kaggle/
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test.csv 
sudo apt-get install unzip
unzip test.csv.zip -d climsim_dataset


-----------------------------------------------------------------------

git clone https://github.com/Joy-Lunkad/climsim-max.git --branch haiku

cd climsim-max/

pip install -r requirements.txt

kaggle -v
gsutil cp gs://us2-climsim/kaggle.json ./
cp kaggle.json ~/.kaggle/

python3 train.py --exp_version 2.0



  gcloud compute tpus queued-resources describe qr-1 \
--project ai-memory \
--zone us-central2-b

gcloud compute tpus queued-resources list --project ai-memory \
--zone us-central2-b

gcloud alpha compute tpus queued-resources create qr-1 \
--node-id us-node-1 \
--project ai-memory \
--zone us-central2-b \
--accelerator-type v4-32 \
--runtime-version tpu-ubuntu2204-base

gcloud alpha compute tpus queued-resources create pre-qr-1 \
--node-id pre-us-node-1 \
--project ai-memory \
--zone us-central2-b \
--accelerator-type v4-32 \
--runtime-version tpu-ubuntu2204-base \
--best-effort

ssh-keygen -f ~/.ssh/google_compute_engine
# ssh into it

# Run git clone and cd into climsim-max

eval "$(ssh-agent -s)"
ssh-add /home/joylunkad/.ssh/google_compute_engine

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
--zone=us-central2-b --worker=all --command="pip install \
--upgrade 'jax[tpu]>0.3.0' \
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"


gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="git clone https://github.com/Joy-Lunkad/climsim-max.git --branch haiku"



gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd climsim-max/ && pip install -r requirements.txt"

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="echo 'export PATH=\$PATH:/home/joylunkad/.local/bin' >> ~/.bashrc"

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="bash -l -c 'source ~/.bashrc && kaggle --version'"


kill $(ps -ef | grep train | awk '{print $2}' | head -n 2)

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="gsutil cp gs://us2-climsim/kaggle.json ./ && cp kaggle.json ~/.kaggle/"


gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd climsim-max/ && git pull"

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd climsim-max/ && python3 train.py --exp_version 4.0"

gcloud compute tpus tpu-vm ssh pre-us-node-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="kill $(ps -ef | grep train | awk '{print $2}' | head -n 2)"
