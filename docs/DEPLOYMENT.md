# CodeMind Deployment Guide

## Local Development
```bash
pip install -r requirements.txt
python codemind.py generate-key
python codemind.py serve
```

## Docker
```bash
cd deploy/docker
docker-compose up --build
```

## Kubernetes
```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
```

## Linux System Service
```bash
sudo cp deploy/systemd/codemind.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable codemind
sudo systemctl start codemind
sudo systemctl status codemind
```

## Training on Kaggle (Free GPU)
1. Go to kaggle.com > New Notebook
2. Enable GPU T4 x2 in Session Options
3. Run:
```python
!git clone https://github.com/0324080078-cyber/CodeMind-LLM.git
%cd CodeMind-LLM
!pip install -r requirements.txt -q
!python codemind.py train
```
