# nlp
This is a pipeline for document classification that have shown effictive in multiple instajces, including a threat intelligence pipeline of some 1m text ducuments daily (to address the ranking problem using labeled data)

Would love to share a presentation on the topic, please contact me at ofer.rahat@tikalk.com

Need to download glove.6B.50d.txt for this to work

https://www.kaggle.com/desiredewaele/sentiment-analysis-on-imdb-reviews

##########
# DevOps #
##########
(1) start gc instance of type e2-medium (2 vCPUs, 4 GB memory) at us-west1-a
    https://console.cloud.google.com/compute/instances?authuser=1&folder=&organizationId=&project=sentiment-262109&instancessize=50
    with ubuntu 18.04 (lsb_release -a)
(2) install docker and docker-compuse 
    https://phoenixnap.com/kb/install-docker-compose-ubuntu
(3) add /usr/local/bin to path
    PATH="$PATH:/usr/local/bin"
(4) edit docker-compose.yml
(5) sudo sysctl -w vm.max_map_count=262144
(6) To open the port for local connection, you can either open directly on the instance (least secure),
    use nginx server to relay from port 5601 to 80 (add password), or use the following ssh tunnel:
    ssh -L 5601:localhost:5601 ofer_rahat_tikalk_com@34.82.219.114 -N
