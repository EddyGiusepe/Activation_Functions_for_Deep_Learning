Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Estudo baseado em [Jason Brownlee PhD](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)


# Activation functions for Deep Learning

As funções de ativação são uma parte crítica do projeto de uma Rede Neural.

A escolha da função de ativação na `camada oculta` controlará quão bem o modelo de rede aprende o conjunto de dados de Treinamento. A escolha da função de ativação na camada de saída definirá o tipo de previsão que o modelo pode fazer.

Como tal, uma escolha cuidadosa da função de ativação deve ser feita para cada projeto de rede neural de aprendizado profundo.

Neste tutorial, você descobrirá como escolher funções de ativação para modelos de `Rede Neural`.

Aqui aprenderemos:

* que as funções de ativação são uma parte fundamental do projeto de rede neural.

* que a função de `ativação padrão` moderna para camadas ocultas é a [Função ReLU](https://www.deeplearningbook.com.br/funcao-de-ativacao/).

* A função de ativação para `camadas de saída` depende do tipo de problema de previsão.



# Funções de ativação

Uma função de ativação em uma rede neural define como a soma ponderada da entrada é transformada em uma saída de um nó ou nós em uma camada da rede.

Às vezes, a função de ativação é chamada de `transfer function` (função de transferência). Se a faixa de saída da função de ativação for limitada, ela pode ser chamada de `squashing function` (função de esmagamento). Muitas funções de ativação são não-lineares e podem ser chamadas de `não-linearidade` na camada ou no projeto da rede.

A escolha da função de ativação tem um grande impacto na capacidade e desempenho da Rede Neural, <font color="orange">e diferentes funções de ativação podem ser usadas em diferentes partes do modelo</font>.

Tecnicamente, a função de ativação é usada dentro ou após o processamento interno de cada nó na rede, embora as redes sejam projetadas para usar a mesma função de ativação para todos os nós em uma camada.

**Uma rede pode ter três tipos de camadas**: camadas de entrada (<font color="orange">input layers</font>) que recebem entrada bruta do domínio, camadas ocultas (<font color="orange">hidden layers</font>) que recebem entrada de outra camada e passam a saída para outra camada e camadas de saída (<font color="orange">output layers</font>) que fazem uma previsão.

<font color="yellow">Todas as camadas ocultas normalmente usam a mesma função de ativação</font>. A camada de saída normalmente usará uma função de ativação diferente das camadas ocultas e depende do tipo de previsão exigida pelo modelo.

As funções de ativação também são tipicamente **diferenciáveis**, o que significa que a derivada de primeira ordem pode ser calculada para um determinado valor de entrada. Isso é necessário uma vez que as redes neurais são tipicamente treinadas usando o algoritmo de <font color="orange">backpropagation</font> (retropropagação) do erro que requer a derivada do erro de previsão para atualizar os pesos do modelo.

Existem muitos tipos diferentes de funções de ativação usadas em redes neurais, embora talvez apenas um pequeno número de funções seja usado na prática para camadas ocultas e de saída.

![](https://neigrando.files.wordpress.com/2022/03/neuronio-e-rede-neural.png)

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVU7r6rEjZKIIwuYuhLLO4-Xg9dQ3iYK8N7bRj5LqOGIZ34W2rhR9uK_RWMAWQdjhCyA&usqp=CAU)

# Ativação para camadas ocultas

Uma `camada oculta` em uma rede neural é uma camada que recebe entrada de outra camada (<font color="orange">como outra camada oculta ou uma camada de entrada</font>) e fornece saída para outra camada (<font color="orange">como outra camada oculta ou uma camada de saída</font>).

Uma camada oculta não contata diretamente os dados de entrada ou produz saídas para um modelo, pelo menos em geral.

<font color="yellow">Uma rede neural pode ter zero ou mais camadas ocultas</font>.

Normalmente, uma **Função de ativação não-linear diferenciável** é usada nas camadas ocultas de uma rede neural. Isso permite que o modelo aprenda funções mais complexas do que uma rede treinada usando uma **Função de ativação linear**.


Existem talvez três funções de ativação que você pode considerar para uso em camadas ocultas; eles são:

* Ativação Linear Retificada (<font color="yellow">ReLU</font>)

* Logística (<font color="yellow">Sigmóide</font>)

* Tangente Hiperbólica (<font color="yellow">Tanh</font>)

<font color="red">Esta não é uma lista exaustiva de funções de ativação usadas para camadas ocultas, mas elas são as mais usadas.</font>

Vamos dar uma olhada em cada um por sua vez.


# Função de Ativação de Camada Oculta ReLU

A <font color="yellow">função de ativação linear retificada</font>, ou função de ativação **ReLU**, é talvez a função mais comum usada para camadas ocultas.

É comum porque é simples de implementar e eficaz para superar as limitações de outras funções de ativação anteriormente populares, como <font color="yellow">Sigmoid</font> e <font color="yellow">Tanh</font>. Especificamente, é menos suscetível a [vanishing gradients](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/) que impedem o treinamento de modelos profundos, embora possa sofrer outros problemas como unidades saturadas ou “mortas”.

A [Função ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) é calculada da seguinte forma:

$$max(0.0, ~x)$$

Isso significa que, se o valor de entrada ($x$) for negativo, um valor $0.0$ será retornado, caso contrário, o valor será retornado. 

Ao usar a função **ReLU** para camadas ocultas, é uma boa prática usar uma inicialização de peso “He Normal” ou “He Uniform” e escalar os dados de entrada para o intervalo $0-1$ (normalizar) antes do treinamento. <font color="yellow">Ver gráfico no arquivo Python!</font>


# Função de ativação de camada oculta sigmóide

A <font color="yellow">função de ativação sigmóide</font> também é chamada de <font color="yellow">função logística</font>.

É a mesma função usada no algoritmo de <font color="yellow">classificação de regressão logística</font>.

A função recebe qualquer valor real como entrada e produz valores na faixa de $0$ a $1$. Quanto maior a entrada (`mais positivo`), mais próximo o valor de saída estará de $1.0$, enquanto quanto menor a entrada (`mais negativo`), mais próximo o valor da saída estará de $0.0$.

A função de ativação sigmóide é calculada da seguinte forma:

$$\frac{1}{1 + e^{-x}}$$

Onde $e$ é uma constante matemática, que é a base do Logaritmo Natural. <font color="yellow">Ver gráfico no arquivo Python!</font>



# Função de ativação de camada oculta Tanh

A <font color="yellow">função de ativação da tangente hiperbólica</font> também é chamada simplesmente de função Tanh (também `tanh` e `TanH`).

É **muito semelhante** à função de ativação sigmóide e até tem a mesma forma de "S".

A função recebe qualquer valor real como entrada e produz valores na faixa de $-1$ a $1$. Quanto maior a entrada (mais positiva), mais próximo o valor da saída estará de $1$, enquanto quanto menor a entrada (mais negativa), mais a saída será para $-1$.

A função de ativação **Tanh** é calculada da seguinte forma:

$$\frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Onde $e$ é uma constante matemática que é a base do logaritmo natural. Ao usar a função TanH para camadas ocultas, é uma boa prática usar uma inicialização de peso “Xavier Normal” ou “Xavier Uniform” (também referida como inicialização de Glorot, nomeada para Xavier Glorot) e escalar os dados de entrada para o intervalo $-1$ a $1$ (por exemplo, o alcance da função de ativação) antes do treinamento. <font color="yellow">Ver gráfico no arquivo Python!</font>


# Como escolher uma função de ativação de camada oculta

Uma rede neural quase sempre terá a mesma função de ativação em todas as camadas ocultas.

É muito incomum variar a função de ativação por meio de um modelo de rede.

Tradicionalmente, a função de ativação sigmóide era a função de ativação padrão na década de $1990$. Talvez entre meados e final da década de $1990$ a $2010$, a função Tanh era a função de ativação padrão para camadas ocultas.

Tanto a função sigmoide quanto a função Tanh podem tornar o modelo mais suscetível a problemas durante o treinamento, através do chamado problema de **vanishing gradients**.

A função de ativação usada em camadas ocultas normalmente é escolhida com base no tipo de arquitetura da rede neural. Modelos modernos de redes neurais com arquiteturas comuns, como `MLP` e `CNN`, farão uso da função de ativação `ReLU`, ou extensões.


As redes recorrentes ainda costumam usar funções de ativação `Tanh` ou `sigmoid`, ou mesmo ambas. Por exemplo, o `LSTM` normalmente usa a ativação `Sigmoid` para conexões recorrentes e a ativação `Tanh` para saída.

* <font color="pink">Multilayer Perceptron (MLP): função de ativação `ReLU`</font>.

* <font color="pink">Rede Neural Convolucional (CNN): Função de ativação `ReLU`</font>.

* <font color="pink">Rede Neural Recorrente: Função de ativação `Tanh` e/ou `Sigmoid`</font>

Se você não tiver certeza de qual função de ativação usar para sua rede, experimente algumas e compare os resultados.

# Ativação para Camadas de Saída

<font color="pink">A camada de saída é a camada em um modelo de rede neural que emite diretamente uma previsão.</font>

Todos os modelos de rede neural `feed-forward` têm uma camada de saída.

Existem talvez três funções de ativação que você pode considerar para uso na camada de saída; eles são:

* Linear
* Logística (Sigmoide)
* Softmax

Esta não é uma lista exaustiva de funções de ativação usadas para camadas de saída, mas elas são as mais comumente usadas.

# Função de Ativação de Saída Linear

A função de ativação linear também é chamada de “identidade” (multiplicada por $1$) ou `sem ativação`.

Isso ocorre porque a função de ativação linear não altera a soma ponderada da entrada de forma alguma e, em vez disso, retorna o valor diretamente.

# Função de ativação de saída Softmax

A função softmax gera um vetor de valores que somam $1$ que pode ser `interpretado como probabilidades` de associação de classe.

Está relacionado à [função argmax](https://machinelearningmastery.com/argmax-in-machine-learning/) que gera um $0$ para todas as opções e $1$ para a opção escolhida. Softmax é uma versão “mais suave” de argmax que permite uma saída semelhante à probabilidade de uma função do vencedor leva tudo.

Como tal, a entrada para a função é um vetor de valores reais e a saída é um vetor de mesmo comprimento com valores que somam $1$ como probabilidades.


# Como escolher uma função de ativação de saída

Você deve escolher a função de ativação para sua **camada de saída** com base no tipo de problema de previsão que está resolvendo.

Especificamente, o tipo de variável que está sendo prevista.

<font color="red">Por exemplo:</font> você pode dividir os problemas de previsão em dois grupos principais, prevendo uma variável categórica ( <font color="yellow">classificação</font>) e prevendo uma variável numérica (<font color="yellow">regressão</font>).

Se o seu problema for um problema de regressão, você deve usar uma função de ativação linear.

## Regressão : 
Um nó, ativação linear.

Se o seu problema for um problema de classificação, `existem três tipos principais de problemas de classificação` e cada um pode usar uma função de ativação diferente.

Prever uma probabilidade não é um problema de regressão; é `classificação`. Em todos os casos de classificação, seu modelo preverá a probabilidade de associação de classe (por exemplo, probabilidade de que um exemplo pertença a cada classe) que você pode converter em um rótulo de classe nítida por arredondamento (para `sigmoid`) ou argmax (para `softmax`).

* Se houver duas classes mutuamente exclusivas (<font color="pink">classificação binária</font>), sua camada de saída terá um nó e uma função de `ativação sigmóide` deve ser usada. 

* Se houver mais de duas classes mutuamente exclusivas (<font color="pink">classificação multiclasse</font>), sua camada de saída terá um nó por classe e uma `ativação softmax` deverá ser usada. 

* Se houver duas ou mais classes mutuamente inclusivas (<font color="pink">classificação multilabel</font>), sua camada de saída terá um nó para cada classe e uma função de `ativação sigmóide` será usada.




| Tipo de tarefa           | Função de Ativação a usar           | 
| :---                     |     :---:                           | 
| Classificação Binária    | Um nó, ativação sigmóide            | 
| Classificação multiclasse| Um nó por classe, ativação softmax  | 
| Classificação Multilabel | Um nó por classe, ativação sigmóide |





Thanks God!
