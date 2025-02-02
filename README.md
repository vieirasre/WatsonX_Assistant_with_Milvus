# WatsonX_Assistant_with_Milvus

Como construir um WatsonX Assistant que utilize NeuralSeek com Milvus Vector Database.

### 1 - Criação do Virtual Server

 - Faça login na sua conta do IBM Cloud [aqui](https://cloud.ibm.com/).
 - Na barra de pesquisa, digite "Virtual Servers" e selecione a opção "Virtual Servers" nos resultados.
 - Na página de Virtual Servers, você encontrará várias opções de servidores virtuais. *A sua localização precisa ser DALLAS*
 - Vá para "Image and Profile" e clique em "Change image" para buscar por "ubuntu" e escolha "22.04 LTS Jammy Jellyfish Minimal Install" e clique em "save".
 - Clique na opção de servidor virtual escolhida para começar a configurá-lo.
 - Na página de configuração, você será solicitado a fornecer vários detalhes como localização, CPU, memória, armazenamento e sistema operacional. Escolha "bx2-2x8".
 - Crie uma chave SSH com o nome "pem_ibmcloud" e faça o download.
 - Complete as opções restantes de configuração como default.
 - Clique no botão "Create" para criar a instância.
 - *Crie um IP flutuante que será usado como IP público.*

> **Nota**: Se a chave for baixada como `minha-chave.prv`, renomeie para `minha-chave.pem`.

### 2 - Conecte-se ao Virtual Server

Você precisará ter o OpenSSH Client atualizado no seu Windows. Consulte os links abaixo para mais informações:

- [Install OpenSSH for Windows](https://learn.microsoft.com/pt-br/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui#install-openssh-for-windows)
- [Conectando-se a uma instância do VSI no IBM Cloud](https://cloud.ibm.com/docs/vpc?topic=vpc-vsi_is_connecting_windows)

Conecte-se usando o comando:

```sh
ssh -i C:\caminho\para\minha_chave.pem ubuntu@<ip_flutuante>
```
 
Atualize o sistema:
 ```sh
 sudo apt-get update
 ```
 
Verifique se o Python 3 está instalado. Se não estiver, instale:
 ```sh
 python3 --version
 ```
 ```sh
 sudo yum install python3
 ```

Remova pacotes desnecessários:
```sh
sudo apt autoremove
```

Verifique se o Git está instalado. Se não estiver, instale:
```sh
git --version
```
```sh
sudo yum install git
```

Instale o pymilvus:
```sh
pip3 install numpy pymilvus

```

### Step 3 - Instalação do Milvus

Baixe e instale o Milvus:
```sh
wget https://github.com/milvus-io/milvus/releases/download/v2.3.7/milvus_2.3.7-1_amd64.deb
sudo apt-get update
sudo dpkg -i milvus_2.3.7-1_amd64.deb
sudo apt-get -f install
```
Reinicie o Milvus:
```sh
sudo systemctl restart milvus
```
Verifique o status do Milvus: 
```sh
sudo systemctl status milvus
```


Verifique se a porta está aberta (use o IP privado):
```sh
telnet <ip_privado> 19530
```
nesse passo o ip usado é o privado!!


### 4 - Criar Token de acesso no HuggingFace
Siga as instruções do HuggingFace para criar um token de acesso.

### 5 - Clonagem do repositório do GitHub

Clone o repositório:
```sh
git clone https://github.com/vieirasre/WatsonX_Assistant_with_Milvus.git
```
```sh
cd WatsonX_Assistant_with_Milvus
```
Instale as dependências:
```sh
pip install -r requirements.txt
```

### 6 - Configuração do HuggingFace
Exporte o token de acesso para o ambiente:
```sh
export HUGGINGFACEHUB_API_TOKEN=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
```

### 7 - Criar e Inserir Dados na Coleção
Crie uma coleção:
```sh
python collection_maker.py
```

Insira os dados na coleção: 
```sh
python index-milvus-comentado.py
```


### 8 - criação do usuario e senha no milvus

```
```
Passo 1: Criar o Arquivo de Configuração
Primeiro, você precisa criar o arquivo de configuração server_config.yaml no seu servidor virtual onde o Milvus está rodando.

Conecte-se ao seu servidor virtual (parece que você já está conectado).

Crie um novo arquivo de configuração. Você pode usar um editor de texto como nano ou vi. Por exemplo, para usar o nano:

```
```
Cole o seguinte conteúdo no arquivo, ajustando a senha conforme necessário:

```

```

Salve e feche o arquivo (Ctrl + O para salvar no nano, Ctrl + X para sair).

Passo 2: Executar o Milvus com Docker
Agora você precisa executar o Milvus com o Docker, montando o arquivo de configuração criado. Certifique-se de que o caminho para o arquivo de configuração está correto.

No terminal do seu servidor, execute o comando Docker, substituindo ~/server_config.yaml pelo caminho para o seu arquivo de configuração:
```

```







































