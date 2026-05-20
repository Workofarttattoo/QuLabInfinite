import { useState, useEffect } from 'react';
import { RouterProvider } from 'react-router';
import { router } from './routes';
import { BootSequence } from './components/BootSequence';

export default function App() {
  const [bootComplete, setBootComplete] = useState(false);
  const [hasBooted, setHasBooted] = useState(false);

  // Check if user has already seen boot sequence in this session
  useEffect(() => {
    const booted = sessionStorage.getItem('qulab_booted');
    if (booted === 'true') {
      setBootComplete(true);
      setHasBooted(true);
    }
  }, []);

  const handleBootComplete = () => {
    sessionStorage.setItem('qulab_booted', 'true');
    setBootComplete(true);
  };

  if (!bootComplete && !hasBooted) {
    return <BootSequence onComplete={handleBootComplete} />;
  }

  return <RouterProvider router={router} />;
}