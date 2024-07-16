# WatsonX_Assistant_with_Milvus
Como construir um WatsonX Assistant que utilize NeuralSeek com Milvus Vector Database.
Esse é um guia ppara poder utilizar o tutorial abaixo, tive várias dificuldades então estou anotando aqui tudo o que fiz para conseguir segui-lo.

Siga o tutorial:
 https://github.com/ruslanmv/Watsonx-Assistant-with-Milvus-as-Vector-Database/blob/master/README.md

### Step 1 - Creation of the Virtual Server
location: DALLAS
CRIA UMA IP FLUTUANTE QUE TU VAI USAR ELA como ip pública

se a chave for baixada como minha-chave.prv é só mudar o nome para minha-chave.pem

### Step 2 - Connection to the server
 Você vai ter que ter o Client OpenSSH no seu windows atualizado então segue aqui alguns links pra você ler:
 
 https://learn.microsoft.com/pt-br/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui#install-openssh-for-windows

 https://cloud.ibm.com/docs/vpc?topic=vpc-vsi_is_connecting_windows

 e me conectei usando:
 ```
 ssh -i C:\caminho\para\minha_chave.pem ubuntu@ip_flutuante
 ```
 
 atualizei o sistema:
 ```
 sudo apt-get update
 ```
 
 tem que ter Python 3, se não tiver instala:
 ```
 python3 --version
 ```
 ```
 sudo yum install python3
 ```

Verifica se tem git e se não tiver instala :)
```
git --version
```
```
sudo yum install git
```
e segue o tutorial a partir da parte de instalação do milvus :)

```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.7/milvus_2.3.7-1_amd64.deb
sudo apt-get update
sudo dpkg -i milvus_2.3.7-1_amd64.deb
sudo apt-get -f install
```


```
sudo systemctl restart milvus
```

```
sudo systemctl status milvus
```



depois de instalar *Python 3.10* eu fiz um 
```
sudo apt autoremove
```
na parte de verificar instalação use pip3 no lugar de pip
```
pip3 install numpy pymilvus

```

e quando for usar python como comando use python3.
```
python3 hello_milvus.py
```

Verificação de porta:
nesse passo o ip usado é o privado!!

para colocar a rodas app.py:
clone o repositório do ruslan, instale as dependências
```
git clone https://github.com/ruslanmv/Watsonx-Assistant-with-Milvus-as-Vector-Database.git
cd Watsonx-Assistant-with-Milvus-as-Vector-Database
pip install -r requirements.txt
cd container-api
pip install -r requirements.txt
```
Crie uma Watson Machine Learning Service Instance - https://eu-de.dataplatform.cloud.ibm.com/docs/content/DO/WML_Deployment/WMLServiceInstance.html?context=analytics



DEFINA:
 - WATSON_APIKEY - você vai criar uma seguindo: IBM Cloud > IAM > Chaves de API > criar chave. 
https://dataplatform.cloud.ibm.com/docs/content/wsj/admin/admin-apikeys.html?context=wx&audience=wdp

 - PROJECT_ID - entra no projeto > gerenciar > detalhes > id do projeto
```
export WATSONX_APIKEY=your_api_key
export PROJECT_ID=your_project_id
```

meu código:
git clone https://github.com/vieirasre/WatsonX_Assistant_with_Milvus.git
cd WatsonX_Assistant_with_Milvus
pip install -r requirements.txt



















































