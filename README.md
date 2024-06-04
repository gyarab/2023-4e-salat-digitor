> [!WARNING]
> **Tento repositář byl archivován.**
> Ve vývoji tohoto programu budu pokračovat v mém osobním repositáři <a href="https://github.com/ondrej-salat/digitor">https://github.com/ondrej-salat/digitor</a>

# Digitor [<i><a href="https://gyarab.github.io/2023-4e-salat-digitor/main.pdf">Online PDF dokumentace</a></i>]

Tento projekt slouží k vytváření a trénování jednoduchých (plně propojených) neuronových sítí. Konkrétní
model `digitor.json`
byl natrénován pro klasifikaci rukou napsaných cifer.
Tento model slouží především jako demonstrace, že program je schopen vytvářet a úspěšně natrénovat různé modely.

## Instalace

```sh
git clone https://github.com/gyarab/2023-4e-salat-digitor.git
cd 2023-4e-salat-digitor
sh install.sh
```

## Spuštění

```sh
cd build
./digitor [flags] [arguments]
```

#### Možnosti:

* `[flags]`
    * `-t`: Program se spustí v módu učení
    * `-n`: Program místo načítání dat z již existujícího souboru vytvoří nový

* `[arguments]`
   ```shell
  ./digitor -t -n <neurony> <iterace> <rychlost_učení> <aktivace> <počet_dávek> <počet_dat>
  ``` 
   ```shell
  ./digitor -t <jméno_soubor> <počet_iterací> <rychlost_učení> <počet_dávek> <počet_dat>
  ```
  ```shell
  ./digitor -n <neurony> <aktivace>
  ``` 
  ```shell
  ./digitor <jméno_souboru> 
  ```