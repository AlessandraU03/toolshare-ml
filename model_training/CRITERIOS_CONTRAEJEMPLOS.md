# Mineria de contraejemplos para romper sesgos residuales

El modelo de desgaste (nuevo / uso_moderado / viejo_desgastado) todavia comete
dos tipos de error sistematico, detectados al analizar los errores de
validacion:

## Patron A — "foto de stock/profesional" -> predice "nuevo" aunque este
desgastada

Fotos de catalogo/stock (fondo blanco o negro liso, bien iluminadas, objeto
centrado, a veces con marca de agua tipo Getty/Dreamstime) se predicen como
"nuevo" incluso cuando la herramienta tiene oxido visible, grietas, o pátina
marcada. El modelo asocia "estilo de foto profesional" con "producto nuevo".

**Buscar:** imagenes con ese estilo de foto (fondo solido/estudio, buena
iluminacion, aspecto de foto de stock) donde la herramienta en si SI muestra
desgaste real y claro: oxido naranja-cafe, grietas, mango roto/astillado,
suciedad muy incrustada, pintura descarapelada.

**Si la encuentras:** clase_corregida = "uso_moderado" o "viejo_desgastado"
segun el desgaste real (usa los mismos criterios de siempre: oxido/grietas
claros = viejo_desgastado; desgaste leve sin oxido = uso_moderado).

**Ejemplo real confirmado por el usuario:** foto tipo producto de un martillo
(cabeza metalica, mango naranja/negro, insignia circular "Made in USA") sobre
fondo de concreto/piedra — el modelo la predijo "nuevo" con alta confianza,
pero el usuario que probo la app la considera claramente usada. Presta
atencion especial a: cara de golpeo de la cabeza con micro-rayones o brillo
desigual (no perfectamente pulida), mango con leve decoloracion o brillo
disparejo, logos/etiquetas ligeramente desvanecidos. Ese tipo de desgaste
SUTIL en fotos de estilo producto es justo lo que el modelo se sigue
saltando — no esperes solo oxido naranja obvio, tambien cuenta como
contraejemplo de Patron A si el desgaste es real pero discreto.

## Patron B — "fondo/superficie sucia o desordenada" -> predice desgaste
aunque la herramienta este nueva

Herramientas fotografiadas sobre superficies visiblemente sucias, manchadas,
con textura rugosa (mesa de taller manchada, concreto, aserrin, tierra,
cartón) o en contexto de uso activo (aserrin volando, manos sucias) se
predicen como "uso_moderado" o "viejo_desgastado" aunque la herramienta en si
este limpia, sin oxido, sin rayones ni daño.

**Buscar:** imagenes con fondo/superficie sucio, texturizado, de taller real,
o contexto de uso activo, donde la herramienta en si SI se ve nueva/cuidada:
sin oxido, sin grietas, pintura intacta, mango sin desgaste visible (puede
tener polvo superficial leve, eso no cuenta como desgaste).

**Si la encuentras:** clase_corregida = "nuevo".

## Que NO copiar

La gran mayoria de las imagenes NO son contraejemplos utiles — sáltalas:
- Fotos de catalogo con herramienta genuinamente nueva sobre fondo limpio
  (el caso normal, ya sobra en el dataset).
- Fotos casuales con herramienta genuinamente desgastada sobre fondo sucio
  (tambien el caso normal, ya sobra).
- Cualquier imagen ambigua donde no estes seguro del patron.

Solo interesan las combinaciones "cruzadas" (estilo de foto que no coincide
con el estado real de la herramienta) — esas son las que le faltan al
modelo para dejar de usar el fondo como atajo.
