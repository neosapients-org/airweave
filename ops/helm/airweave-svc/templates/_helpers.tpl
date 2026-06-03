{{/*
Expand the name of the chart.
*/}}
{{- define "airweave-svc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "airweave-svc.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "airweave-svc.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "airweave-svc.labels" -}}
helm.sh/chart: {{ include "airweave-svc.chart" . }}
{{ include "airweave-svc.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "airweave-svc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "airweave-svc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "airweave-svc.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "airweave-svc.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Common StatefulSet template for stateful infra components (postgres, redis, vespa, temporal).
Usage: {{ include "airweave-svc.statefulset" (dict "root" . "name" "postgres" "config" .Values.postgresql) }}
*/}}
{{- define "airweave-svc.statefulset" -}}
{{- $root := .root -}}
{{- $name := .name -}}
{{- $cfg := .config -}}
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ $name }}
  labels:
    {{- include "airweave-svc.labels" $root | nindent 4 }}
    app.kubernetes.io/component: {{ $name }}
  annotations:
    argocd.argoproj.io/sync-wave: "1"
spec:
  serviceName: {{ $name }}
  replicas: 1
  selector:
    matchLabels:
      {{- include "airweave-svc.selectorLabels" $root | nindent 6 }}
      app.kubernetes.io/component: {{ $name }}
  template:
    metadata:
      labels:
        {{- include "airweave-svc.selectorLabels" $root | nindent 8 }}
        app.kubernetes.io/component: {{ $name }}
    spec:
      securityContext:
        {{- toYaml $cfg.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ $name }}
          image: {{ $cfg.image }}
          {{- with $cfg.command }}
          command:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with $cfg.args }}
          args:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with $cfg.env }}
          env:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          ports:
            - containerPort: {{ $cfg.port }}
          {{- with $cfg.livenessProbe }}
          livenessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with $cfg.readinessProbe }}
          readinessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          resources:
            {{- toYaml $cfg.resources | nindent 12 }}
          volumeMounts:
            - name: data
              mountPath: {{ $cfg.dataMount }}
            {{- with $cfg.extraVolumeMounts }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
      {{- with $cfg.extraVolumes }}
      volumes:
        {{- toYaml . | nindent 8 }}
      {{- end }}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: {{ $cfg.storageClassName }}
        resources:
          requests:
            storage: {{ $cfg.storage }}
{{- end }}

{{/*
Common ClusterIP Service for infra components.
Usage: {{ include "airweave-svc.infraservice" (dict "root" . "name" "postgres" "port" 5432) }}
*/}}
{{- define "airweave-svc.infraservice" -}}
{{- $root := .root -}}
{{- $name := .name -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ $name }}
  labels:
    {{- include "airweave-svc.labels" $root | nindent 4 }}
    app.kubernetes.io/component: {{ $name }}
spec:
  type: ClusterIP
  selector:
    {{- include "airweave-svc.selectorLabels" $root | nindent 4 }}
    app.kubernetes.io/component: {{ $name }}
  ports:
    - port: {{ .port }}
      targetPort: {{ .port }}
{{- end }}
