# RecSys Challenge RL

Deverá ser criado um modelo de Deep Learning para geração de embeddings de itens que será utilizado em um sistema de recomendação por similaridade. O objetivo é que seja utilizado o máximo de features e combinações possíveis (a criatividade é o limite) na criação da representação do item (embedding), de modo a otimizar uma função de recomendação baseada em similaridade a partir dos embeddings criados.

Ref: https://docs.google.com/document/d/1RxrNXsfzq4WlpQBSOXuczdHUXt5SXrQ6hrwilwqKIQg/edit?usp=sharing

## Dataset
https://www.yelp.com/dataset

O datastet da Yelp é composto por 6 arquivos contendo diferentes informações:
* business.json: Contém dados de negócios, incluindo dados de localização, atributos e categorias.
* review.json: Contém dados completos do texto da resenha, incluindo o user_id que escreveu a resenha e o business_id para o qual a resenha foi escrita.
* user.json: User data including the user's friend mapping and all the metadata associated with the user.
* checking.json: Check-ins em um negócio.
* tip.json:  Dicas escritas por um usuário em um negócio. As dicas são mais curtas do que as avaliações e tendem a transmitir sugestões rápidas.
* photo.json: Contains photo data including the caption and classification (one of "food", "drink", "menu", "inside" or "outside").

O objetivo do trabalho é que sejam criados os embeddings dos bussiness, que no caso são os restaurantes a serem recomendados a partir da similaridade. Todas as informações no dataset podem ser utilizadas para criação do embedding, embora nem todas sejam úteis.

## Baseline

O Base line consiste em utilizzar um modelo de linguagem do huggingface para extrair os embeddings dos nomes dos restaurantes utilizando a biblioteca https://github.com/huggingface/transformers

### Transforma o Dataset `.json` em `.csv`

```bash
python json_to_csv_converter.py data/yelp_dataset/yelp_academic_dataset_business.json
```

### Exporta os embeedings utilizando o modelo 

script:
```
extract_embedding.py <model_base> <csv_file> <output_path> 

Params:

model_base: The model base to extract the embeddings.
csv_file: The csv file to extract the embeddings.
output_path: The output path to save the embeddings and metadata.
```

Example: 
```bash
python baseline/extract_embedding.py bert-base-uncased data/yelp_dataset/yelp_academic_dataset_business.csv data/output/
```

Após extrair os embeddings utilizando o script `extract_embedding.py` serão criados os arquivos:

- **embeddings.txt**:  Contém apenas os embeddings dos itens
- **metadados.csv**: Contém todos os metadados utilizados junto com o business_id para identificar o embedding do item. Os dois arquivos devem estar na mesma ordem pra dar match.

## Avaliação

Example:

```bash
python evaluation/evaluation.py data/baseline/embeddings.txt data/baseline/metadados.csv
```