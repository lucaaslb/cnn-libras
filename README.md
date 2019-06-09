# Redes Neurais Convolucionais

Esse projeto tem como objetivo gerar um classificador com Redes Neurais Convolucionais para reconhecimento de gestos do alfabeto em LIBRAS. 

#Deep Learning #LIBRAS #InteligenciaArtificial #CNN #Python3 <br> <br>

```
INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT 
```

#### Requirements
> Criar um ambiente no Anaconda: 
- conda create --name nome_ambiente 
>Ativar o ambiente e instalar as bibliotecas: 
- source activate nome_ambiente
- conda install -c anaconda tensorflow 
- conda install -c conda-forge keras 
- conda install -c anaconda scikit-learn 
- conda install -c conda-forge matplotlib
- conda install -c anaconda pydot

#### Structure

> Dataset/ - Contem o dataset e scripts para gerar novas imagens <br>
> Main/ <br>
>> cnn/ <br>
>>> __ init __.py  - Estrutura das camadas da CNN <br>
>> train.py  - Execução de treinamento importando a estrutura da cnn. <br>
>> app.py - Teste do modelo para reconhecimento em real-time com OpenCV e o modelo de CNN treinado.<br>
<br>

> Models/ - Contem graficos, imagens de modelos e modelos treinados <br> 

> logs/  - Logs de execução com informações de epocas, validação e sumario <br> 

<br>

#### References

CNN: http://cs231n.github.io/convolutional-networks/ 

Documentação Keras: https://keras.io/


