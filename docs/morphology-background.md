# Rerefons: morfologia indoeuropea, regularitat i arbitrarietat

> Per què esperem que un morfema es comporti com una **direcció vectorial
> constant**, i per què la lingüística històrica prediu que aquesta regularitat
> existeix però **no és perfecta**. Aquest document emmarca la mètrica de
> *consistència de direcció* ([`methodology.md`](methodology.md) §5) en la
> teoria morfològica, i motiva l'anàlisi de regularitat de
> [`findings.md`](findings.md) §10.

---

## 1. L'estructura del morfema indoeuropeu

Una paraula en una llengua indoeuropea s'analitza canònicament en **arrel**
(morfema lèxic) + **sufix derivatiu** (forma un tema) + **desinència flexiva**.
El català n'és hereu directe: `ràpid`(arrel) → `ràpida`(tema femení) →
`ràpida·ment`(adverbi); `cant·`(arrel) → `cant·em`(1a pl. present).

La derivació indoeuropea era fonamentalment **sufixal**, i aquest patró es
manté a les llengües filles. La hipòtesi composicional que provem —que el
significat/funció d'una paraula complexa es deriva sistemàticament de les seves
parts— és la formalització moderna d'aquesta estructura clàssica
(Lazaridou et al. 2013; Marelli & Baroni 2015).

## 2. Per què esperar regularitat: el morfema com a direcció

Si un sufix aporta sempre la mateixa operació, els desplaçaments
`v(derivat) − v(base)` haurien de ser **paral·lels** entre tots els parells
d'una família — una sola direcció lineal a l'espai latent. És la mateixa lògica
que la "direcció de gènere" de Bolukbasi et al. (2016) i la *linear
representation hypothesis* de Park, Choe & Veitch (2024). La nostra
**consistència de direcció** mesura literalment aquest paral·lelisme: 1,0 =
morfema perfectament regular; 0 = sense operació compartida.

## 3. Per què la regularitat NO és perfecta: tres forces erosionadores

La lingüística històrica prediu desviació respecte a l'1,0, i n'explica la
causa:

1. **Canvi fonètic regular → irregularitat morfològica (paradoxa de
   Sturtevant).** Els neogramàtics (Osthoff & Brugmann 1878) van postular que
   el canvi fonètic és **exceptionless**; però en aplicar-se cegament trenca la
   uniformitat dels paradigmes. *"El canvi de so és regular i crea
   irregularitat; l'analogia és irregular i crea regularitat."* L'al·lomorfia
   catalana (plural `-s`/`-os`, `feliç→feliços`) n'és el residu.

2. **Gramaticalització: bleaching semàntic + erosió fonètica.** Un mot lèxic
   esdevé afix perdent significat i forma (Hopper & Traugott; Bybee, Perkins &
   Pagliuca). El cas de llibre és el nostre protagonista: l'adverbialitzador
   `-ment` ve del llatí **MENTE** ('amb la ment/esperit'). Com més vell i
   gramaticalitzat és un sufix, més difusa esdevé la seva contribució
   composicional.

3. **Lexicalització / idiosincràsia.** Molts derivats es congelen amb un
   significat propi (*educació* no és només 'l'acte d'educar'). Marelli & Baroni
   ho formulen com la tensió central de la morfologia derivativa: **reconciliar
   la idiosincràsia morfològica amb la regularitat semàntica**.

## 4. Arbitrarietat al mot, sistematicitat a la categoria

Dingemanse et al. (2015) i Monaghan, Christiansen & Fitneva (2011) mostren a
gran escala que l'**arbitrarietat domina al nivell del mot individual**, però hi
ha **sistematicitat al nivell de la categoria gramatical** — precisament on viu
la morfologia. Això és el que fa que la pregunta sigui empírica i no retòrica:
*queda prou sistematicitat de categoria perquè el morfema sigui una direcció, o
l'arbitrarietat acumulada ja l'ha dissolt?*

## 5. La predicció quantitativa: productivitat ⇒ regularitat

Baayen (1992 i seg.) dona mesures de **productivitat morfològica** per afix. La
generalització rellevant: els processos **flexius** (plural, gènere, persona
verbal) són altament productius i regulars; molts processos **derivatius**
estan més subjectes a lexicalització i deriva. D'aquí la hipòtesi falsable amb
les nostres dades:

> **H:** la consistència de direcció nadiua és **més alta en la flexió que en
> la derivació**, i decreix amb el grau de gramaticalització/lexicalització del
> sufix.

L'anàlisi de [`findings.md`](findings.md) §10 prova aquesta hipòtesi amb les 10
famílies morfològiques catalanes. La bibliografia completa és a
[`references.md`](references.md).
