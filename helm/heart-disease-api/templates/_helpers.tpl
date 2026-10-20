{{/* Common name and labels, reused by every template in this chart. */}}

{{- define "heart-api.name" -}}
{{ .Release.Name }}
{{- end }}

{{- define "heart-api.labels" -}}
app: {{ .Release.Name }}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "heart-api.selectorLabels" -}}
app: {{ .Release.Name }}
{{- end }}
