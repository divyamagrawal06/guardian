import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaView, Text, View, StyleSheet, Button } from 'react-native';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Companion App</Text>
        <Text style={styles.subtitle}>Expo Go (managed) — TypeScript</Text>
        <Button title="Example button" onPress={() => {}} />
      </View>
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff'
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 16
  }
});
