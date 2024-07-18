# WatsonX_Assistant_with_Milvus
Como construir um WatsonX Assistant que utilize NeuralSeek com Milvus Vector Database.

### 1 - Criação do Virtual Server
location: DALLAS
CRIA UMA IP FLUTUANTE QUE TU VAI USAR ELA como ip pública

se a chave for baixada como minha-chave.prv é só mudar o nome para minha-chave.pem

### 2 - Conecte-se ao Virtual Server
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

Depois de instalar o *Python3* eu fiz um 
```
sudo apt autoremove
```

Verifica se tem git e se não tiver instala :)
```
git --version
```
```
sudo yum install git
```

Instalação do pymilvus:
```
pip3 install numpy pymilvus

```

### Step 3 - Instalação do Milvus

```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.7/milvus_2.3.7-1_amd64.deb
sudo apt-get update
sudo dpkg -i milvus_2.3.7-1_amd64.deb
sudo apt-get -f install
```
Restart do milvus:
```
sudo systemctl restart milvus
```
Verifica o status do milvus: 
```
sudo systemctl status milvus
```


Verificação de porta:
```
telnet <ip_privado> 19530
```
nesse passo o ip usado é o privado!!


### 4 - Criar Token de acesso no HuggingFace

### 5 - Clonagem do repositório do GitHub


```
git clone https://github.com/vieirasre/WatsonX_Assistant_with_Milvus.git
```
```
cd WatsonX_Assistant_with_Milvus
```
```
pip install -r requirements.txt
```

exporte para o ambiente o token de acesso feito no hugging face:

```
export HUGGINGFACEHUB_API_TOKEN=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
```


Crie uma coleção:
```
python collection_maker.py
```

Insira os dados na coleção: 
```
python index-milvus-comentado.py
```











































