# Quijote_GPT

This repository contains the work dedicated to the creation and training of a Transformer model capable of generating "Don Quixote-like" text. Using as train and evaluation instances the two parts of the book *Don Quixote*, from the spanish author Miguel de Cervantes.

The transformer architecture is created using the knowledge learned during my Master's degree in Computer Vision and also after visualizing the  [following YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy, the very-well known computer scientist. On the other hand, the Tokenizer architecture is created after watching the tutorial found in [this YouTube video](https://youtu.be/zduSFxRajkE?feature=shared), which was also made by Andrej Karpathy.



## Tokenization

The Tokenization process is done using a *Regex Tokenizer*, implemented by myself in [this folder](main/Tokenizer.py), and the creating and learning process can be seen in the *Tokenizer* folder. 

The Tokenizer uses the following GPT-4 Regex pattern:

```python
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

On the other hand, until now, I have just used one configuration of the Tokenizer, for which I chose the train it until *200 merges* were done. Instead of using the entire book of *Don Quixote*, I wanted to use different datasets for the Tokenizer's training and the Transformer's training, so I used another very well known book of the Spanish Literature: *Cien Años de Soledad*, by Gabriel García Márquez. 

On the other hand, I have used some special tokens which serve to identify the beginning and end of each chapter and each chapter title:

```python

tokenizer.register_special_tokens({
        "<|beginchaptername|>": 456,
        "<|endchaptername|>":   457,
        "<|beginchapter|>":     458,
        "<|endchapter|>":       459,
    })

```


## Data preprocessing

I used UTF-8 to encode each character, which ended up creating some issues, since some characters which are typical in spanish have no encoding. All the accent marks: á, é, í, ó and ú, and their corresponding uppercases; the inverse exclamation and interrogation marks (¡ and ¿) also have no encoding; and some other special characters, such as ñ or ü.
So, in the case of the inverse interrogation and exclamation marks, they have both been taken out from the datasets. And all the characters with accent marks and other special letters have been converted to regular letters, for instance: á --> a and ñ --> n.


## Datasets

The used datasets can be found in the *databases* folder. When creating the Transformer and the Tokenizer some english datasets where used, which also can be found in the specified folder. But, for the Tokenizer training the novel [Cien años de soledad](databases/cien_anos_de_soledad.txt) was used (200 merges), and for training the Decoder-Transformer the two parts (enclosed in one file) of the novel [Don Quixote](databases/quijote.txt) where used.


## Quijote-GPT training

When training the Decoder, different learning rate values where used: 5e-5, 3e-4 and 1e-3. The decoded has been trained using 100.000 steps, but, as it can be seen in the image below, the best validation loss is obtained quite fast.

![loss plots](images/Quijote_GPT_losses.png)



## SOME GENERATIONS:

Even though the number of chapters is not too high and hence it is quite difficult to learn the '<|beginchapter|><|beginchaptername|> *Chapter number* - *Chapter name* <|endchaptername|><|endchapter|>' structure, in some cases it is done, and quite correctly:



````md
<|beginchapter|>
<|beginchaptername|>
Capitulo XI
De lo que le sucedio a don Quijote con el valiente hijo de don Clavileno y se vituviese de la
ver por la duquesa o princesa.
<|endchaptername|>

Dejamos el gran grandes altituya de la aventura de los desta tres puntos la hermosa
que esta a la redonda Evolvertida de la Trifaldi, como el de la princesa, yo la llevara, o la
respuesta presente; a quien desde la guerra se viene arroja en mitad de aquella gente pudiencia al
cabo, con una alguna grande aventura que esta aqui cerca, donde fue la que el tenia cuando llamaba, de
aquellos lumbra y desenvainando la hacer tan a los caballeros; y quien se lamenta la causa no habian
dermitidos del dia.
 Marcela, adonza Rodriguez, y al parecer dijo que con mas alegrile que se mostro lien
ordenadas e intimas y regalo de mis amores, y cuando Ni Sancho le pusieron desta
maneracion, el cura y el que en aquel ejercicio las habia conocido, y el otro le respondio que se llamaba lo
primero.
-No -dijo don Quijote- no era razon, sino que estaba puesta en una piedra, sino los aventamientos que
se le asen de mi la docena, y que le hayan dejado de mi reino y por mi senor. Veis
habian leido que en hala, y Camila debe de estar en la primera su estado, es el pueblo de la
caballeriza, y los hace Pedro y el arcable lasciverias del Pamas, el auto de Toledo Basilio de
Acaparcio, y los dejoos era bauscal del jardin, que ellos quiso darse sobre sus panos a su casa, que no
fueran de contra las
senas. Iba por entretendiendo don Quijote y ventura acogerse y descrear los intereses son tonto los
desenganos de lo que ella se ha visto; y no se, como dicen los tora batalla, ni acaso por todos los
manjares.
En resolucion, la armo del pueblo y se lo dijo muy bien en ello:
-Aqui no hay alguno la senora, que el paje que encierra en esta carta historia pienso
llor y los bajos, que por tan mal se burluce a senor don Quijote, porque el no salia a uno, disparada y con el
mayordomo del duque; y lo que la duquesa, que de se les dio un de boto, y lo que el recibio le
habia de acepto y conto todo lo apunto, y dijole Sancho:
-Sepaos, senor Caballero de la Triste Figura, que si vuestra merced no aciertara, que
tal vez primera o no le alzo.
Y el brazo impidio uno de los cuadrilleros con sus cuchos, que habian
tambecido la pelea con el confirmiento de los duques, y como ya habia de ser luego en cosa de
recer, excepto las de la Villalva. Estaba Sancho Panza-, adonde o por si; y, asi, a
descosar, Sancho mio, no le faltaran al oficio que tan alegresentada. Pero, en que me toquenosas de gobernador,
veinte y consumirse, de mal aquella ganador de mi patria, y que yo soy don Quijote con el,
tan ajenme en escureceros de tu vida, que, si alguna mujer bien en ello se hace por la andar por
habermoso la justicia, que dicen que la falta es batalla por medioso
vuestro despues que su gobernado acierto y mi deseo veo, y por lo que toca a oparme la voz temia, o a lo
que escucha tenia, o lleno de prosiguio, diciendo:
-Ninguno se yo le oye que, cuando responde a la ambicion de enoje, pues sera bien que la
parte que este gigante se me vaya a la musica.
La senora, que por este rusticado tenia la promesa de tu cuerpo, y don Quijote
atentando por aquellas manos, le dijo:
-Por cierto, hermano, vuesas mercedes, sen estas dos doncellas que redunstantes, y capondiran unos monos decendidos por
balles palabras.
-Tambien debe de ser el enemigo -dijo el cura-, pero el regidor le han dado este libro.
-Adonde vas a el -dijo el cura-, que ya habia de ser llamar al pastor Quijotiz; y este Hicrifaltan serencio del dia, parece que
tenia de ser asismas y mas que aquel momento que a sus criados; pero el quien era, son arzodose, tan
uno puno como se lo habia parecido.
Al castillo del duque y la duquesa, que dejole referir a cucarle y, al parecer, de la cosa torres; y
lo suplicaba, que los infinitos pertaban de los secretos y siempre uno de un reino de a otrocientos escritos y
desabidores de letras; y si el ventero se llevaban con el aposento, donde quedo don Quijote
pasmado, sino las cuales vino a los caballeros por la reja de su valentia, con una maravillosa y luego la
abrosa.
-Eso no, senor -respondio Sancho-, ni en mitad del pecho mio; pero sabraste has de ahora,
me digo, que de mis cinco y carnes.
-Oh miserables tan cristianos de amojos, y, asi, sin duda, debiendo una de las mas
ajer sus puertas de ovejas, y sin sufrir liciosa suerte, que a por su intencion le
hallaban por do asir de noche, en proseguir sanecies y en que saludio; y, asi, con facilidad
podria er que fuesen celada y con ellas habia pasado en peligro con todo esto.
-Oh cuento no -dijo la Dolorida-, pero no la dice a ver en todas las cuentas y a todas las
en que veces musicas dellas de mi amo. Marten, pues sabed cosas de memoria, y no las cosas de que en
ellos haremos del de su lugares las bocas del mundo, sino a los hipocritas. Todo lo cual se
daba contende en el suelo, a quien estaba muy quebrada la cabeza; y la una dellas, llego duenas a los lados.
Los de a buscarle; y aun seria bien que uno de ser licenciado nos respondia palabra alguna
parte a la que el gusto que se les ofrecia. Asi que, senor, movido asi de Lotario y Lotario, que dicen:
CA todo esto que los Doce un
 Libro en ella la Blanca Lotario, puesto que fue el de la Blanca Luna que a don Quijote le lacayo la mitad
que habia cuando estaba lavado, aunque le habia paseado la bondad de su parte don Quijote.
Altisidora (en todo el bailla de juzgar la de Sancho Panza, el gobernador
Sancho Panza, de cuya llevar a don Quijote, nuevas del Granada no le declarase su discrecion de los
senores y sentia palafrena de la baja, y aun por todo el cortesia de la gradura de los terminos de
Udo, y no que por eso dejan de serlo.
-Eso juzgo yo -dijo el labrador-, y aunque no lo estaba, ni los doler, que el todos los limites de
las valerosas, porque no se han de entender ni el saberlas que a mi os entienda.
-Tambien lo es, senora ama de mi -dijo Sancho-, y en razon tan discreto, como al mismo, senor, que
no son otro, sino duque, los hechos de ver en una alta.
-No -dijo Sancho Panza-, que yo la llevare yo la bendito de la senora Dulcinea del Toboso, que solo
pecador de mi! Aqui sera poder la virtud que te ha hechizos en el mar Mecamino ahora si de consorte hasta que es antigua de
serviros la que piensa . Por esta seria, o quien la sangreti del bergantia que espero, las cuales bien
en se quitase al cielo, pero nuestro liberal seria alguna barbero, algun dia se por alguna burla que el se lo
prometia.
El licenciado la beberle. Lestaronle Sancho y, asi, le dijo las hechas le agradecian:
-Quese testimose don Quijote, amigo -dijo don Quijote-, y beso te vuestra merced por aqui
el mancebo, tan valiente y tan focido enamorado como don Gaiferos venea en busca y en
su tierra. Esto es lo que se cumplase, aunqueno conservar en la mas ropilla de la ciudad, castillo
Angrimis destos, trajes, que bien se que no se espian a la corte, sinoque algunos humilos de los ojos
pasados siempre los escubros de Utris, atendiendo los caminos y le derribaron y tendieron con gran piedad
naciendo a su recua y cortesia; y unto en estremo adula camisa, pero,
en los libros que, con cuyos muros se puso unos corazones los oidos de los peligros, y aun
por las encima del pcato del Toboso.
Tenia el cura de Leandra, cuya pregunto el barbero y de Esparcia y aun de
su amigo y saco debajo de la venta por ver que por aquella carmilla fuese manifiesto por ella. Nerabia de
poder llamar a la casa de don Diego, por su varia y oficio para gusto de los cuales nombres
que me pasaban, si fue fuesen de rodar en mas promesar de todo el
ganero, como si no fuera por la verdadera reduja de las circunvecinas perpetuo neciales y rostro, y acordais de su
hombre, porque, en priva se acaba los vomidados de un grande arcabo. Y, en fin, se poneda la ansfa a cada paso anos a los
gobernadores della. En este tiempo se le puso el perro y se varala referida y las postruras,
sobre y distintamente, de quien el vomito de ellas me gabajaban y yo le empleaba, especialmente la mohina
Antonomaso». Sentose el caballero, que viendose rompito y y chizo otros dan continentes a mas que a
semejantes pensamientos, no me los llevara la comida, sin que jamas al pellizcar despacio a mi rey
acarejo de beber por ella. Y si estoy senal alguno la verdad otro dia en esto de la tardanza?
-Tambien la dije -dijo a esta sazon el barbero-, y esta presente Cadradillo!
-Quedehaya quince salir de esa ciudad, y aun se debe de ausencia!
-Que me pides arrojaras de los ruintos y arroganteza de su pastres y deste mi libro, y
tanto mas que e
````


Another example:


````md
<|beginchapter|>
<|beginchaptername|>
Capitulo XLVIII
De lo que sucedio a don Quijote con dona Rodriguez, la duena de la duquesa, con otros
<|endchaptername|>

acontecimientos dignos de escritura y de memoria eternacer esta en razon de hidalgo con el rigurero, y de la
cuarta fue del talle de la historia, y aunque el era su culpor y lugar de que el tal le tiene en su casa, que no se
pide de otra cosa, y aun de media noche celada la envidia con la campana de don Quijote y aquel
acomento le vieron, y prosiguiendo con Teneral al vicario con el deseo que
pareciese de alli adelante; tomole don Quijote a lo que pudiese costais. Y tanto aqui
fui el azotare y la cara, por ver si se mostro, con buenas de los mejores que el duque tenia, y
todavia pensaba ya que Camila recebia, porque era la principal de haber del podado disculpa de su
pulso, sino aquella tan inquieta ostrepido corazon, como es nuestro, para lo
maravillanidad de mi senora la duquesa, digna de las ancas rendidades, asperezas y otras mil
nombres: y aun viviria yo, aunque no les aquen este pleito, contandolos no me las
moridasen a la mano Dulcinea, y los magos dellos y cinco dientes, no piensos. Y
para todos el nuevo vientre de ser hasta que tu linda Magalona vida humana corre a su
condicion, y mas que la historia cobrare, y que entonces se muestra en ella cinco capa le
un medio, y con tanta risa como un gran pobre lacayo de buey las riberas.
-Como dices eso? -respondio don Quijote-. No oyes el relinchar de los caballos, el mochacho y
llegiros que tras todo el oro en las nubes del andante, cubierto de los pescadores del
caballo, y los demas que alli se cuentan de don Quijote y aquel salio Ebajo del
cuento iba venida a la suya, y asi como se le vio entrar
en su aposento solo, se luego, con una muy buena gracia de la condicion que el marido pensamiento en
palabra y con que le tratare, y aun si a la buena fe se prometen a su puesto. El
cual, estando en la pendencia a mi mas partida, no fue tan seguidado las armas, subiendo a quien
parosedian en esto el ignorante de su invencion, a los que, a vagarenos a la corte a parte; pero, veis
asi, saltandome descubierta con unas espaldas de cien criadas que le
arrenalles, y le murmurasen de atras, no hay saber de la que mas necesidad.
-Eso juro yo bien -respondio Sancho- se lo que haya Panza moviere a mi Dios, que a mi se me
dierece que es necesidad tienen remedio don Quijote, y asi el diablo, que nos cubre a los turcos, ni pueden responden a
ningun bajel, pues, con su mesma regalo, el cual quiera vez tiene la casada, aunque es su artificio no es
que vos mucho el rostro con pelliroso y corona de la ciudad a desechaba. Diose por
soca su merced que se la era grudito: «Guijote direto -dijo don Quijote-, el nombre podra don
Quijote de la honda, deje pasar, y el aire la gula y la he dado, y el
otro, al primero, y no le alcanza; pero vuesa merced al buen que me ha dicho en esto: «Mirad que me ha de dar gusto a vuestra merced al
senor de las mas invidias que se me da el mundo. Asi que, senora ama: podria ser que hombre juramento es yelmo: «Digo aquel gran don Qoso
caballero andante poetacion de Dios, como tambien me digais a sus solas y a
solechillas en las muesas partes del mundo, y si le hallasen apriesa quiso ir a los alcuzos y al del
temio que habian hecho los encantadores.
-No le fue hicieron para el senor don Quijote -dijo a esta sazon el cura-, que yo se que ha podido
apear con su manda a tus pasados los cuentos y fueren su deseo, porque a pesar de tus
alcanzas no agradecio las que tienen noches hacer ni hablar quien ellos es, y si ello mas creo que ellos fueran
tenian que Angelicase las grandes y liberalidades que su historia; aunque Dios me llaman «Teresa».
-No me paso, doncellas -dijo don Quijote-, y replicandola que los escuderos se miran, y que
el que los buenaran que quieren se usan en la cama, que no se los lleven a los trabajos, a la que no podra
causar quien bienes,de no estudiarlas (puesto que no estuviese) tuvieron los
escuchales, tales, que te fueron a los que los reyes respetos del canonigos y de entre
priscalles sacarse los arboles, dando un tiento de su muerte.
-Senor -dijo don Quijote-, tiempos hay de burlar y tiempos demasia de un alcornoque, y
que se dejen sentir de las que me hacen
de aqui adelante de las dulces ayes de triunfario y trizados a nuestro pueblo y entre los
llegares en que han de parar a otras cosas asperezas? Porque si he tenido vuestra merced tierra en
esta soledad ponga y mejor asno las hidalgas, y es que te vuestra merced las obras que os satisfagan en tal
publico, y tomaremos la mesma vida del rucio un ardite, que la mas del amor se
debe de haber dado las cojas de hacer, de treinta y costa de poner en camino, de todo el
que pie o callarlos la carcel hicieron? Con todo eso -respondio Sancho-, hay tananos puntuales hay en la tierra
hermosura para las republicas, y de los representantes, como ellos traigan y
princesas.
-Senor -dijo Sancho-, tienes algun andadoviza de renegar?
-Calla, no os alla, Sancho -respondio don Quijote-, que yo no se puede entender hablar, se me entiende cuatro
en una torre al suelo.
-Eso sali yo -respondio don Quijote- debe de estar, y aunque para mi alcanzarlo, que me
ha de llevar condi que vuestra merced dice?
-Yo lo hubo y llegado como quien venzaba con la priesa que la ley de
claro en la caballeria que tenia; mas como Sancho, en su ano y costal de Espana, ya
 iba a capa, y que no se le cocia en zaga
 de caballeros valientes. De almas con seguridad de los historiadores y enferos, y que aamas habeis dicho
estan las necedades en bestia, y que esta mala rapecha; la cual, hacedar al reves, que nadie son de
uso de su ventura, hacia la discreta, reina, y al agravio a femencillo, el segundo postrados y
el demas rico que le rindiesen. A lo que dijo le
sal hizo, como digo, sin volver a Sancho:
-Sancho, guia al pavor de Dios y a la vida, que por ser vuestra merced tiene
uno de sobra, y no se ha ofrecido otras muchas, suelen ser mentirosas, solicitamente los
duques naturales son tan promesas, que no tiene las armas de los hermosos
de poetado. Esta  senora mia, en efecto, que veasmado de holgarse y de tomo
sepa como decirla, y no arrogan otra vez trabajo, y advertidme manana,Sancho, que una vez no es muy buena gana por ventura de so sus
malicios o imaginacion. Digo, en fin, alta y todos los del recado estaba otro que se podia componer a
su caballero solicito, le volviese y a ser molido, quedando otro palabras en la primera parte desta historia
que tal trance Altisidora, que tales cosas parge solicitudes y gracias, que no tan presto se puede
dar acomodarme de pocas, por que de mi tengo que valdra a mi senora la duquesa; y mas
todo el de mi amo, que nos hace al caso a la verdad de la historia de aquella mesmo caballeros
andantes: a los pasados, la alegres y los viernos, los premios que esperan baratar los generos de ojos. Decharemos
bueno, por lo que me pregunto que mirase y que no estaba obligado a recostar en ninguna
manera. Franciendo nuestros, que se dice y no hay para que se enviuena ni quiero yo que los deje
hecho?
-No se lo que dijo -replico Sancho-: solo se que los hicistes de la caballeria, puesto que, siendo a
la muerte criada de tanta necha, a tranoche con crea, y una sarpa, en tanto mentecato.
En esta se quieto don Quijote en oyendo el camino y ayunarla asi mando a la polsildea y que
se le ofrecia.
-Cuerpo de tal! -dijo a este intervaliente, la puso de rodillarse el en las espaldas y las demas de Ramon, y estoy yo se de cuando
yo vino a imaginar que se le habeis dado.
-Y tambien decirselo -dijo Sancho- era si amo de no haberme tan secreto.
-Cuerpo de tal! -dijo a este punto el gobierno-. Porque no sera peor ningun desencin, y
vera el buen los agravio que en silencio vivos al suelo, y debeis que se ha de
tomar con tanto rijoso, para meterse con el ir a la orevencion de Verdad, para cu yo se ha tenido en
conocimiento de la desencante que la honestidad de mi senora me da su gusto, que me la ha de dar las muelas y la la
de Trifaldi y convidida en que se consige; y si no, tanto llora y la bendicion de Trifaldin; y torno le dire
tanto, anduve de mucho caso castigolo porque es dinero, del dinero y pasado adelante, la de
admiracion de mi deseo.
-Y yo se quito y le ha dicho que fue la que ha de posar -respondio Clara.
Y por no olvidarme aqui, que el Caballero de los Espejos era escudero de Leon, y cuando dentro apellan y asi
como lo vera con vuestra merced la dama que de tener tan buenos amores toca: salieron bien, y
desi es que luego todo el
````



And there also are some generations which do not have the title referenced correctly, but the text inside is of a quite high quality,
which have ended when generating the *'<|endchapter|>'* token.

````md

<|beginchapter|> el valeroso caballero caballero don Fernor y trayendole de que no
tendran a Fernando el rostro con asaltadas las frutos de los corredores y escriberos con mas
de puertas, con cualquiera calor basion.
Prodeados yo el que quiere calzar de nuevo modo que los contenian el sabio menor mujer
prometido de don Alvaro; porque el que me he ofrecer esta caballeria dice que bien la de
voluntad, y seguramente puede desflo a las
amos a los de bocacios reales, y yo anda en vos y en su esfunda, y tan bien y tantas trabte lecreto y
companeros del mismo lienzo y tanto del vulgo trato como de los despojos de la morirre, ya podia
de quien por ella que los zancados del mayor loco conde nos supo, pasaron aquellos disparates que la
virtud habremos, en dos que al sumeros para el paso.
 Otramo vez digo que decis, pues sois  amor aquella fue Lobres de Amadis,
 que de los que retozadores les malandrines,
 por enamoradas y deseosas obligadas para dar la reja a la nospienda a batalla,
 con otros tres dias hijas que nuestra cortesia nos piden a otra celada, como solamente mi y no
tengo nada.-De en disimondad de vuestra merced -respondio la Trifaldi-, si yo lo dije yo que se decir o no quisiere
cometir, a lo menos Juanto; «vire la mano, por cuyo no macido, que por piensa la verja «de los
Toledo», dejo de ser caballero, o mejor decir, a vista: yo soy mal contino, y soy un poeta Emperador, y
el portcialmente me entiende por vos y excelillinar todos los inconvinentes.
-Senor -respondio el bachiller Carrasco don Quijote-, que si no anda afligido a oro borracho, pero no hay historia
que cuando vay, cuando desdichados del cuerpo de su casa, ni senalarios y tiene alguna, se ha de ver campana
las luciente del jumento rostro, las estruengo, no hubiera univa de apararer en privo en la
ventana, pues venid de sola. A la ventera en breve tresta y casi dandole la noche senal por no cobarrimado
de bruces. En resolucion, tanto, que Tome maravillosame, y vino pagar a buscar por ese mandonela,
por cuya infinita voz primero, que hay parosa iguales tan cierta que cada una en que poder aquel
capicado arbol, estaba mucha contra tormento. Y aunque lo viniese y le preciase, como pudieron
solidose y regalaban de juicio, y dejaronle la camisa, que dejaron la cama del romance, armas por
lo que seis a el don Diego aquella fuese; y a venian esperaron.
Subieron dar al viejo, pues, que alli muchos poseos en varian con en
gramada la bota y, asio del cura y a don Quijote un hombre sus poderosas que hacia basta y un
azotes; y aunque hablando todos los jueces, todos las dos pasados con don Quijote. Don Quijote pendiendo
la mala vista vino, vino los duques que a el labrador el al viento, para conto se puso en breve
negro a la duquesa; y yendo asimismo el suceso y, finalmente, diciendo a dar a todas las
cellas, dijo a su padre:
-Esta el cuento depare seno, que no te des, ni lo sera: es la caza aventura de monterosa.
Desta de nosotras que apuntan pensaba en las de Trancancas manos, y ello sus manos, por
verte que decia delante: quiero decir que el Caballero delero de la MaNCarela, que apocase a este
poseinas su amada y a nuestro dormir, a quien se enamora venia la habian de vener, y que algunos
tumas hacia ellos los subiese, respondio Lotario la senora Dulcinea como pudieron los
duques de los que tuvo el duque quisiera.
Anadis quedose don Quijote, pero los caminantese todos, por entrar
comentos, cuando los vivos a pasaron entre unos amores de la gallarda, y los demas que del Toboso volvia,
se fueron al pobre don Quijote estuvo.Salieronle al visorrey su acrevido y orden con priesa le hablota la
cueva de su retira. Y al pasado don Quijote, se tomola al visior consigo otro con gravisa de
tomarle sin bueno mi ventura, jamas ajenas a que entonces sus primerias. Oyendo lo
que el agujero dio orden campo de palto y que no habia comenzado, y dijo a su Diornanil que
Allevaria dando un corpel de la cabeza doncellas de don Quijote. Hizolo asimesmo la
amisima a la litera y de hacer cuando en alguna parvedad de su senora, volviendo a rendir las yerbolas a
recebien a su casa, digo, sin duda alguna, o castillo o no, que quiso igual, porque aunque no sabe mas
dar
selos, menoscas y do lo ejemplo si su esposa hambre y que pensaba hacer ahora, haga venir a ser
del leones puesto en lo que el fatige, que veemos hecho a la albripacite.
Peneciolense que es esto asi como lo vio encantamento con el espicargos a esta sazon dijo el
verso que dicen «a los dos sentencimientos».
-«Y eso era, el mayor Cance anos -respondio don Quijote-, don Fernando como el de Sancho, el mayor
gobernador historia.
-Si he dicho, bien puede, senor? -respondio el otro-. Porque el que nos matase a vuestra
madrida que, puesto sea albane Altisidora, la cual ya habra un demonio con unos hacha reverentado
las espaces, y los cabellos a tales pies y descubrieron otras venia una vez cerdujeros; y, con todo,
puesto hace con mucho agrado, rompiendo que no se pasa pudiese perder el donaivo, que, por aunque
deste peligro.
 -Bien puede teces haga, senora bien, que no hay tocado de mi por loco.
-Asi es mio, pues la mia -replico don Quijote-. Oyendo eso, senor Sanson Carrastria, que era otro
mayeron, que fue: Pero dispusose el de la Triste Torde Bretana, y esto, llego otro y otro con que
Caco que don Quijote habia hecho la mano de Sancho, pero pensaron que Sancho era de laar Mavela del
mundo, y que todo lo demas dijeron a su amo, porque se lo entregasen, y que el tal, que le
primieron requegidio persigo, sin ser estidos, a la primera lesia
pregunto a ser el que todos los diciplinantes.
Y para que los que les demas tuvieron pasas, y manchose el camino y que cuando don Viva, muelas
Igles y encercada el retablo. Eas los bajos, pues, el amo y vierda por dos lejos, subio sobre quien ya en ella y caballo
don Quijote, que les dio cuenta de reposo que lo de mis perdio.
Viendo esto el v Basilio y robor el cabrero en el suelo, y fue su razonamdo a don Quijote; y
imaginaba que era uno de los tumurjos de la astillana de un alcalz, que entre los dos arboles
a siro y unas maderezadas para regirla: si no hay estado sin remedio y enojo la senora Dulcinea
Panza con el rucio para regocijar a la meson con que buscaba. Finalmente, don Quijote, arene
sles convertido, donde se dio cuentas el cosa de la Salamanca de Quijote.
<|endchapter|>

````



Another example:

````md

<|beginchapter|> y el pensar que el autor Sanson Carrasco cuando el mesmo se aparto de
la Inga artes, y, puesto que por tal mayor que una misma tenia que entre apriesa y de
albordo, y de alli a lo que anoche con voz baja del pobre; y, remirado voces doy orden,
volvieronse a don Quijote y aquel muchacho, que era la verdad de Diego, puestos los ojos en el cristalo
de su boca, como el se iban en su entendimiento; y de todo consintio que le preguntase, si
enhora de que se quejaba esperar en libertad de Sancho no queria, que, a lo que aquellos pregones no
sera bueno empanar a don Quijote, el cual no pequeno se acuerda, sino fingirse con mil
diciplinas, que no hay cosa que llaman «sali»? Tome Cece la homa cruelas, tuerada y si viene vos no con
una fraua o como si soledad, porque es la flojtima del pueblo desto, y que, si no te quedo de contentar
en veinte o aceite, no ha de anadir estos libros. En esto y en estos puestos leales que, para que
salir desde apeantar sin duda ausencia!, asi el amor de la noche, cuando los libros en sus estrechos
bantados, y vienen a los dos en golanda, viendo cuan presentaron a la corte, y aquellos a menos dos
tiempos continuos y no yerran tiene un negocio, y el senor don Quijote, que en nuestros tormentos,
y el senor Sanson de quien os digo a mi Dulcinea del Toboso, que son por esos reales que tendran con
discrecion que caballero, que los curan de Quiteria se entretuviese el barbero y de haberle fin a
tierra gobernador y hace la persona principal cosa con que fue mas promponer del traor buena otro. Halle
a Sevilla que estimamente mil amor».
Tomores dias la apella prendio:
<|endchapter|>

````